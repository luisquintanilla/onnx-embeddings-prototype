using Microsoft.Extensions.AI;
using Microsoft.ML.Tokenizers;
using System.Numerics.Tensors;

namespace CommunityToolkit.Embeddings.Onnx;

/// <summary>MEAI composition over the same public prepare, score and pool stages available to standalone callers.</summary>
public class OnnxEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly bool _ownsEncoder;
    private readonly EmbeddingGeneratorMetadata _metadata;
    private bool _disposed;

    public OnnxEmbeddingGenerator(Tokenizer tokenizer, TokenSequenceOptions sequenceOptions, OnnxTextEncoder encoder,
        PoolingMode pooling, string modelId, int maximumBatchSize = 32, bool ownsEncoder = false)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(sequenceOptions);
        ArgumentOutOfRangeException.ThrowIfLessThan(maximumBatchSize, 1);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (!Enum.IsDefined(pooling)) throw new ArgumentOutOfRangeException(nameof(pooling));
        Tokenizer = tokenizer;
        SequenceOptions = sequenceOptions;
        MaximumBatchSize = maximumBatchSize;
        Encoder = encoder;
        Pooling = pooling;
        ModelId = modelId;
        _ownsEncoder = ownsEncoder;
        _metadata = new EmbeddingGeneratorMetadata("Local ONNX prototype", defaultModelId: modelId, defaultModelDimensions: encoder.Dimensions);
    }

    public Tokenizer Tokenizer { get; }
    public TokenSequenceOptions SequenceOptions { get; }
    public int MaximumBatchSize { get; }
    public OnnxTextEncoder Encoder { get; }
    public PoolingMode Pooling { get; }
    public string ModelId { get; }

    /// <summary>Runs synchronous local inference; no implicit Task.Run, downloads or background work.</summary>
    public virtual Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
        => GenerateCore(values, options, cancellationToken);

    protected Task<GeneratedEmbeddings<Embedding<float>>> GenerateCore(IEnumerable<string> values,
        EmbeddingGenerationOptions? options, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(values);
        cancellationToken.ThrowIfCancellationRequested();
        if (options?.Dimensions is int dimensions && dimensions != Encoder.Dimensions)
            throw new ArgumentException($"This model produces exactly {Encoder.Dimensions} dimensions.", nameof(options));
        if (options?.ModelId is string model && model != ModelId)
            throw new ArgumentException($"This generator only supports '{ModelId}'.", nameof(options));
        if (options?.RawRepresentationFactory is not null || options?.AdditionalProperties is { Count: > 0 })
            throw new NotSupportedException("Raw options and additional properties are not supported. Configure the session explicitly; use E5Purpose for E5 roles.");

        var result = new GeneratedEmbeddings<Embedding<float>>();
        var pending = new List<string>(MaximumBatchSize);
        foreach (string value in values)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ArgumentNullException.ThrowIfNull(value);
            pending.Add(value);
            if (pending.Count == MaximumBatchSize) Flush();
        }
        if (pending.Count > 0) Flush();
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(result);

        void Flush()
        {
            TokenBatch batch = Tokenizer.PrepareBatch(pending, SequenceOptions, MaximumBatchSize, Encoder.RequiresTokenTypeIds, cancellationToken);
            Tensor<float> hidden = Encoder.Score(batch, cancellationToken);
            var buffer = new float[checked(batch.BatchSize * Encoder.Dimensions)];
            EmbeddingPooling.PoolInto(hidden, batch.AttentionMask,
                new TensorSpan<float>(buffer, [batch.BatchSize, Encoder.Dimensions]), Pooling);
            // This private buffer is never recycled or exposed as a mutable tensor. One retained row retains its batch.
            for (int row = 0; row < batch.BatchSize; row++)
                result.Add(new Embedding<float>(buffer.AsMemory(row * Encoder.Dimensions, Encoder.Dimensions)) { ModelId = ModelId });
            pending.Clear();
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (serviceKey is not null) return null;
        if (serviceType == typeof(EmbeddingGeneratorMetadata)) return _metadata;
        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsEncoder) Encoder.Dispose();
        GC.SuppressFinalize(this);
    }
}
