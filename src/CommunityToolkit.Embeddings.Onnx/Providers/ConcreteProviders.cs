using Microsoft.Extensions.AI;
using Microsoft.ML.OnnxRuntime;

namespace CommunityToolkit.Embeddings.Onnx;

public sealed class AllMiniLmL6V2EmbeddingGenerator : OnnxEmbeddingGenerator
{
    public const string SupportedModelId = "sentence-transformers/all-MiniLM-L6-v2";

    public AllMiniLmL6V2EmbeddingGenerator(string assetDirectory, SessionOptions? sessionOptions = null, int maximumBatchSize = 32)
        : base(CreatePreparer(assetDirectory, maximumBatchSize),
            OnnxTextEncoder.Load(Path.Combine(assetDirectory, "model.onnx"), sessionOptions),
            PoolingMode.Mean, SupportedModelId, ownsEncoder: true) { }

    public AllMiniLmL6V2EmbeddingGenerator(string assetDirectory, InferenceSession session, bool ownsSession = false, int maximumBatchSize = 32)
        : base(CreatePreparer(assetDirectory, maximumBatchSize), new OnnxTextEncoder(session, ownsSession: ownsSession),
            PoolingMode.Mean, SupportedModelId, ownsEncoder: true) { }

    private static TextBatchPreparer CreatePreparer(string directory, int batchSize)
    {
        var tokenizer = new BertUncasedTokenizer(Path.Combine(directory, "vocab.txt"));
        return new(tokenizer, new(256, tokenizer.ClassificationTokenId, tokenizer.SeparatorTokenId, tokenizer.PaddingTokenId), batchSize);
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
        : base(CreatePreparer(assetDirectory, purpose, maximumBatchSize),
            OnnxTextEncoder.Load(Path.Combine(assetDirectory, "model.onnx"), sessionOptions),
            PoolingMode.Mean, SupportedModelId, ownsEncoder: true) => Purpose = purpose;

    public E5SmallV2EmbeddingGenerator(string assetDirectory, E5Purpose purpose, InferenceSession session,
        bool ownsSession = false, int maximumBatchSize = 32)
        : base(CreatePreparer(assetDirectory, purpose, maximumBatchSize), new OnnxTextEncoder(session, ownsSession: ownsSession),
            PoolingMode.Mean, SupportedModelId, ownsEncoder: true) => Purpose = purpose;

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

    private static TextBatchPreparer CreatePreparer(string directory, E5Purpose purpose, int batchSize)
    {
        if (!Enum.IsDefined(purpose)) throw new ArgumentOutOfRangeException(nameof(purpose));
        var tokenizer = new BertUncasedTokenizer(Path.Combine(directory, "vocab.txt"));
        return new(tokenizer, new(512, tokenizer.ClassificationTokenId, tokenizer.SeparatorTokenId, tokenizer.PaddingTokenId), batchSize);
    }
}

public sealed class GraniteEmbedding30MEnglishGenerator : OnnxEmbeddingGenerator
{
    public const string SupportedModelId = "ibm-granite/granite-embedding-30m-english";

    public GraniteEmbedding30MEnglishGenerator(string assetDirectory, SessionOptions? sessionOptions = null, int maximumBatchSize = 32)
        : base(CreatePreparer(assetDirectory, maximumBatchSize),
            OnnxTextEncoder.Load(Path.Combine(assetDirectory, "model.onnx"), sessionOptions, outputName: "logits"),
            PoolingMode.Cls, SupportedModelId, ownsEncoder: true) { }

    public GraniteEmbedding30MEnglishGenerator(string assetDirectory, InferenceSession session, bool ownsSession = false, int maximumBatchSize = 32)
        : base(CreatePreparer(assetDirectory, maximumBatchSize), new OnnxTextEncoder(session, outputName: "logits", ownsSession: ownsSession),
            PoolingMode.Cls, SupportedModelId, ownsEncoder: true) { }

    private static TextBatchPreparer CreatePreparer(string directory, int batchSize)
        => new(new Granite30MEnglishTokenizer(Path.Combine(directory, "vocab.json"), Path.Combine(directory, "merges.txt")),
            new(512, 0, 2, 1), batchSize);
}
