using Microsoft.Extensions.AI;
using Microsoft.ML.OnnxRuntime;

namespace CommunityToolkit.Embeddings.Onnx;

public sealed class AllMiniLmL6V2EmbeddingGenerator : OnnxEmbeddingGenerator
{
    public const string SupportedModelId = "sentence-transformers/all-MiniLM-L6-v2";

    public AllMiniLmL6V2EmbeddingGenerator(string assetDirectory, SessionOptions? sessionOptions = null, int maximumBatchSize = 32)
        : base(CreateTokenizer(assetDirectory, maximumBatchSize), new(256, 101, 102, 0),
            OnnxTextEncoder.Load(Path.Combine(assetDirectory, "model.onnx"), sessionOptions),
            PoolingMode.Mean, SupportedModelId, maximumBatchSize, ownsEncoder: true) { }

    public AllMiniLmL6V2EmbeddingGenerator(string assetDirectory, InferenceSession session, bool ownsSession = false, int maximumBatchSize = 32)
        : base(CreateTokenizer(assetDirectory, maximumBatchSize), new(256, 101, 102, 0), new OnnxTextEncoder(session, ownsSession: ownsSession),
            PoolingMode.Mean, SupportedModelId, maximumBatchSize, ownsEncoder: true) { }

    private static BertUncasedTokenizer CreateTokenizer(string directory, int batchSize)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        return new BertUncasedTokenizer(Path.Combine(directory, "vocab.txt"));
    }
}

public enum E5Purpose { Query, Document }

public readonly record struct E5Input(string Text, E5Purpose Purpose);

public static class E5Text
{
    /// <summary>Always prepends the selected task prefix, without trimming or stripping caller text.</summary>
    public static string Format(string text, E5Purpose purpose)
    {
        ArgumentNullException.ThrowIfNull(text);
        return purpose switch
        {
            E5Purpose.Query => "query: " + text,
            E5Purpose.Document => "passage: " + text,
            _ => throw new ArgumentOutOfRangeException(nameof(purpose))
        };
    }
}

public sealed class E5SmallV2EmbeddingGenerator : OnnxEmbeddingGenerator
{
    public const string SupportedModelId = "intfloat/e5-small-v2";

    public E5SmallV2EmbeddingGenerator(string assetDirectory, E5Purpose purpose,
        SessionOptions? sessionOptions = null, int maximumBatchSize = 32)
        : base(CreateTokenizer(assetDirectory, purpose, maximumBatchSize), new(512, 101, 102, 0),
            OnnxTextEncoder.Load(Path.Combine(assetDirectory, "model.onnx"), sessionOptions),
            PoolingMode.Mean, SupportedModelId, maximumBatchSize, ownsEncoder: true) => Purpose = purpose;

    public E5SmallV2EmbeddingGenerator(string assetDirectory, E5Purpose purpose, InferenceSession session,
        bool ownsSession = false, int maximumBatchSize = 32)
        : base(CreateTokenizer(assetDirectory, purpose, maximumBatchSize), new(512, 101, 102, 0), new OnnxTextEncoder(session, ownsSession: ownsSession),
            PoolingMode.Mean, SupportedModelId, maximumBatchSize, ownsEncoder: true) => Purpose = purpose;

    public E5Purpose Purpose { get; }

    public override Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(values);
        return GenerateCore(values.Select(text => E5Text.Format(text, Purpose)), options, cancellationToken);
    }

    /// <summary>Explicit per-item purpose for mixed query/document batches; order is unchanged.</summary>
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<E5Input> values,
        EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(values);
        return GenerateCore(values.Select(input => E5Text.Format(input.Text, input.Purpose)), options, cancellationToken);
    }

    private static BertUncasedTokenizer CreateTokenizer(string directory, E5Purpose purpose, int batchSize)
    {
        if (!Enum.IsDefined(purpose)) throw new ArgumentOutOfRangeException(nameof(purpose));
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        return new BertUncasedTokenizer(Path.Combine(directory, "vocab.txt"));
    }
}

public sealed class GraniteEmbedding30MEnglishGenerator : OnnxEmbeddingGenerator
{
    public const string SupportedModelId = "ibm-granite/granite-embedding-30m-english";

    public GraniteEmbedding30MEnglishGenerator(string assetDirectory, SessionOptions? sessionOptions = null, int maximumBatchSize = 32)
        : base(CreateTokenizer(assetDirectory, maximumBatchSize), new(512, 0, 2, 1),
            OnnxTextEncoder.Load(Path.Combine(assetDirectory, "model.onnx"), sessionOptions, outputName: "logits"),
            PoolingMode.Cls, SupportedModelId, maximumBatchSize, ownsEncoder: true) { }

    public GraniteEmbedding30MEnglishGenerator(string assetDirectory, InferenceSession session, bool ownsSession = false, int maximumBatchSize = 32)
        : base(CreateTokenizer(assetDirectory, maximumBatchSize), new(512, 0, 2, 1), new OnnxTextEncoder(session, outputName: "logits", ownsSession: ownsSession),
            PoolingMode.Cls, SupportedModelId, maximumBatchSize, ownsEncoder: true) { }

    private static Granite30MEnglishTokenizer CreateTokenizer(string directory, int batchSize)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        return new(Path.Combine(directory, "vocab.json"), Path.Combine(directory, "merges.txt"));
    }
}
