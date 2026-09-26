using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;

namespace CommunityToolkit.Embeddings.Onnx;

/// <summary>A validated token-level ONNX encoder. A supplied session is borrowed unless ownership is explicit.</summary>
public sealed class OnnxTextEncoder : IDisposable
{
    private readonly InferenceSession _session;
    private readonly bool _ownsSession;
    private readonly string[] _outputs;
    private bool _disposed;

    public OnnxTextEncoder(InferenceSession session, int dimensions = 384,
        string outputName = "last_hidden_state", bool ownsSession = false)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentOutOfRangeException.ThrowIfLessThan(dimensions, 1);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputName);
        // Ownership transfers only after validation succeeds.
        ValidateInputs(session);
        if (!session.OutputMetadata.TryGetValue(outputName, out var output) ||
            !output.IsTensor || output.ElementType != typeof(float) || output.Dimensions.Length != 3 ||
            (output.Dimensions[2] >= 0 && output.Dimensions[2] != dimensions))
            throw new ArgumentException($"Output '{outputName}' must be a Float32 token tensor [batch, sequence, {dimensions}], not a pooled embedding.");
        _session = session;
        _ownsSession = ownsSession;
        _outputs = [outputName];
        Dimensions = dimensions;
        RequiresTokenTypeIds = session.InputMetadata.ContainsKey("token_type_ids");
    }

    /// <summary>Loads only a local file. Options are borrowed for construction and never disposed by this method.</summary>
    public static OnnxTextEncoder Load(string modelPath, SessionOptions? sessionOptions = null,
        int dimensions = 384, string outputName = "last_hidden_state")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        if (!File.Exists(modelPath)) throw new FileNotFoundException("A local ONNX model file is required.", modelPath);
        InferenceSession session;
        if (sessionOptions is null)
        {
            using var defaults = new SessionOptions();
            session = new InferenceSession(modelPath, defaults);
        }
        else session = new InferenceSession(modelPath, sessionOptions);
        try { return new OnnxTextEncoder(session, dimensions, outputName, ownsSession: true); }
        catch { session.Dispose(); throw; }
    }

    public int Dimensions { get; }
    public bool RequiresTokenTypeIds { get; }

    /// <summary>Returns an owned managed [batch, sequence, hidden] tensor; all native input/output handles are disposed before returning.</summary>
    public Tensor<float> Score(TokenBatch batch, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(batch);
        cancellationToken.ThrowIfCancellationRequested();
        if (batch.BatchSize == 0) return Tensor.CreateFromShape<float>([0, 0, Dimensions]);
        if (RequiresTokenTypeIds && !batch.HasTokenTypeIds)
            throw new ArgumentException("This graph requires token_type_ids.", nameof(batch));
        foreach (var input in _session.InputMetadata)
            CheckBatchShape(input.Key, input.Value.Dimensions, batch);

        long[] shape = [batch.BatchSize, batch.SequenceLength];
        using var ids = OrtValue.CreateTensorValueFromMemory(batch.IdBuffer, shape);
        using var masks = OrtValue.CreateTensorValueFromMemory(batch.MaskBuffer, shape);
        using var types = RequiresTokenTypeIds ? OrtValue.CreateTensorValueFromMemory(batch.TypeBuffer!, shape) : null;
        var inputs = new Dictionary<string, OrtValue> { ["input_ids"] = ids, ["attention_mask"] = masks };
        if (types is not null) inputs.Add("token_type_ids", types);
        using var options = new RunOptions();
        using var registration = cancellationToken.Register(() => options.Terminate = true);
        try
        {
            using var outputs = _session.Run(options, inputs, _outputs);
            cancellationToken.ThrowIfCancellationRequested();
            OrtValue output = outputs[0];
            var info = output.GetTensorTypeAndShape();
            if (info.Shape.Length != 3 || info.Shape[0] != batch.BatchSize ||
                info.Shape[1] != batch.SequenceLength || info.Shape[2] != Dimensions)
                throw new InvalidOperationException($"Output '{_outputs[0]}' did not return [batch, sequence, {Dimensions}].");
            return Tensor.Create(output.GetTensorDataAsSpan<float>().ToArray(), [batch.BatchSize, batch.SequenceLength, Dimensions]);
        }
        catch (OnnxRuntimeException error) when (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException("ONNX inference was cancelled.", error, cancellationToken);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsSession) _session.Dispose();
    }

    private static void ValidateInputs(InferenceSession session)
    {
        if (!session.InputMetadata.ContainsKey("input_ids") || !session.InputMetadata.ContainsKey("attention_mask"))
            throw new ArgumentException("The graph must declare input_ids and attention_mask.");
        foreach (var input in session.InputMetadata)
        {
            if (input.Key is not ("input_ids" or "attention_mask" or "token_type_ids"))
                throw new ArgumentException($"Unsupported graph input '{input.Key}'.");
            if (!input.Value.IsTensor || input.Value.ElementType != typeof(long) || input.Value.Dimensions.Length != 2)
                throw new ArgumentException($"Input '{input.Key}' must be an Int64 tensor [batch, sequence].");
        }
    }

    private static void CheckBatchShape(string name, int[] shape, TokenBatch batch)
    {
        if ((shape[0] >= 0 && shape[0] != batch.BatchSize) || (shape[1] >= 0 && shape[1] != batch.SequenceLength))
            throw new ArgumentException($"Batch [{batch.BatchSize}, {batch.SequenceLength}] is incompatible with input '{name}' [{string.Join(", ", shape)}].");
    }
}
