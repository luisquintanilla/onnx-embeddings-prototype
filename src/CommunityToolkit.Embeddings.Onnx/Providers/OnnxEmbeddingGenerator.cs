using Microsoft.Extensions.AI;

namespace CommunityToolkit.Embeddings.Onnx;

/// <summary>MEAI composition over the same public prepare, score and pool stages available to standalone callers.</summary>
public class OnnxEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly bool _ownsEncoder;
    private readonly EmbeddingGeneratorMetadata _metadata;
    private bool _disposed;

    public OnnxEmbeddingGenerator(TextBatchPreparer preparer, OnnxTextEncoder encoder,
        PoolingMode pooling, string modelId, bool ownsEncoder = false)
    {
        ArgumentNullException.ThrowIfNull(preparer);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (!Enum.IsDefined(pooling)) throw new ArgumentOutOfRangeException(nameof(pooling));
        Preparer = preparer;
        Encoder = encoder;
        Pooling = pooling;
        ModelId = modelId;
        _ownsEncoder = ownsEncoder;
        _metadata = new EmbeddingGeneratorMetadata("Local ONNX prototype", defaultModelId: modelId, defaultModelDimensions: encoder.Dimensions);
    }

    public TextBatchPreparer Preparer { get; }
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
        var pending = new List<string>(Preparer.MaximumBatchSize);
        foreach (string value in values)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ArgumentNullException.ThrowIfNull(value);
            pending.Add(value);
            if (pending.Count == Preparer.MaximumBatchSize) Flush();
        }
        if (pending.Count > 0) Flush();
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(result);

        void Flush()
        {
            TokenBatch batch = Preparer.Prepare(pending, Encoder.RequiresTokenTypeIds, cancellationToken);
            float[] hidden = Encoder.Score(batch, cancellationToken);
            float[][] vectors = EmbeddingPooling.Pool(hidden, batch, Encoder.Dimensions, Pooling);
            foreach (float[] vector in vectors) result.Add(new Embedding<float>(vector) { ModelId = ModelId });
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
