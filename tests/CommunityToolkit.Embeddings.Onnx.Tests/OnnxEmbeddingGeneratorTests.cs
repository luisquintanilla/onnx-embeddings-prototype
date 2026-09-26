using CommunityToolkit.Embeddings.Onnx;
using Microsoft.Extensions.AI;
using Microsoft.ML.Tokenizers;
using System.Runtime.InteropServices;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

public sealed class OnnxEmbeddingGeneratorTests
{
    private static Tokenizer Tokenizer(Func<string, IReadOnlyList<int>>? encode = null)
        => TokenizerExtensionsTests.Tokenizer(encode ?? (text => [int.Parse(text)]));
    private static TokenSequenceOptions Options() => new(8, 1, 2, 0);

    [Fact]
    public async Task Generate_CompletesSynchronouslyChunksAndPreservesOrder()
    {
        using var session = TestAssets.Session("required_types");
        using var encoder = new OnnxTextEncoder(session, 3);
        var calls = new List<string>();
        int callerThread = Environment.CurrentManagedThreadId;
        using var generator = new OnnxEmbeddingGenerator(Tokenizer(text =>
        {
            Assert.Equal(callerThread, Environment.CurrentManagedThreadId);
            calls.Add(text);
            return [int.Parse(text)];
        }), Options(), encoder, PoolingMode.Mean, "contract", maximumBatchSize: 2);
        // Iterator cannot request row 3 until the first batch was actually prepared.
        IEnumerable<string> Input()
        {
            yield return "3"; yield return "6";
            Assert.Equal(new[] {"3", "6"}, calls);
            yield return "9"; yield return "12"; yield return "15";
        }
        var task = generator.GenerateAsync(Input());
        Assert.True(task.IsCompletedSuccessfully);
        var results = await task;
        Assert.Equal(5, results.Count);
        Assert.Equal(new[] {"3","6","9","12","15"}, calls);
        for (int i = 0; i < 5; i++)
        {
            float meanId = i + 2;
            double norm = Math.Sqrt(meanId * meanId + 1);
            Assert.Equal("contract", results[i].ModelId);
            Assert.Equal(3, results[i].Vector.Length);
            Assert.InRange(Math.Abs(results[i].Vector.Span[0] - meanId / norm), 0, 1e-6);
            Assert.InRange(Math.Abs(results[i].Vector.Span[1] - 1 / norm), 0, 1e-6);
            Assert.Equal(0, results[i].Vector.Span[2]);
        }
        Assert.Empty(await generator.GenerateAsync([]));
    }

    [Fact]
    public async Task Generate_ValidatesFixedOptionsAndExposesMetadata()
    {
        using var session = TestAssets.Session();
        using var encoder = new OnnxTextEncoder(session, 3);
        using var generator = new OnnxEmbeddingGenerator(Tokenizer(), Options(), encoder, PoolingMode.Mean, "contract", maximumBatchSize: 2);
        var metadata = Assert.IsType<EmbeddingGeneratorMetadata>(generator.GetService(typeof(EmbeddingGeneratorMetadata)));
        Assert.Equal("contract", metadata.DefaultModelId);
        Assert.Equal(3, metadata.DefaultModelDimensions);
        Assert.Equal("Local ONNX prototype", metadata.ProviderName);
        Assert.Same(generator, generator.GetService(typeof(IEmbeddingGenerator<string, Embedding<float>>)));
        Assert.Same(generator, generator.GetService(typeof(OnnxEmbeddingGenerator)));
        Assert.Null(generator.GetService(typeof(EmbeddingGeneratorMetadata), "key"));
        Assert.Null(generator.GetService(typeof(string)));
        Assert.Throws<ArgumentNullException>(() => generator.GetService(null!));
        var result = await generator.GenerateAsync(["3"], new() { Dimensions = 3, ModelId = "contract", AdditionalProperties = [] });
        Assert.Equal("contract", Assert.Single(result).ModelId);
        Assert.InRange(Math.Abs(result[0].Vector.Span[0] - 2 / Math.Sqrt(6)), 0, 1e-6);
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(["3"], new() { Dimensions = 4 }));
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(["3"], new() { ModelId = "other" }));
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["3"], new() { AdditionalProperties = new() { ["purpose"] = "Query" } }));
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["3"], new() { RawRepresentationFactory = _ => new object() }));
        await Assert.ThrowsAsync<ArgumentNullException>(() => generator.GenerateAsync(null!));
        await Assert.ThrowsAsync<ArgumentNullException>(() => generator.GenerateAsync([null!]));
    }

    [Fact]
    public async Task Generate_CancellationIsObservedThroughEnumerationAndBetweenBatches()
    {
        using var session = TestAssets.Session();
        using var encoder = new OnnxTextEncoder(session, 3);
        int tokenized = 0, enumerated = 0;
        using var generator = new OnnxEmbeddingGenerator(Tokenizer(_ => { tokenized++; return [3]; }), Options(), encoder, PoolingMode.Mean, "contract", maximumBatchSize: 2);
        using var cts = new CancellationTokenSource(); cts.Cancel();
        IEnumerable<string> Before()
        {
            enumerated++; yield return "3";
        }
        await Assert.ThrowsAsync<OperationCanceledException>(() => generator.GenerateAsync(Before(), cancellationToken: cts.Token));
        Assert.Equal(0, enumerated); Assert.Equal(0, tokenized);
        using var between = new CancellationTokenSource();
        IEnumerable<string> Between()
        {
            yield return "3"; yield return "6";
            Assert.Equal(2, tokenized);
            between.Cancel(); yield return "9";
            throw new InvalidOperationException("Enumeration should stop at cancellation.");
        }
        await Assert.ThrowsAsync<OperationCanceledException>(() => generator.GenerateAsync(Between(), cancellationToken: between.Token));
        Assert.Equal(2, tokenized);
        using var end = new CancellationTokenSource();
        IEnumerable<string> AtEnd()
        {
            yield return "3"; yield return "6"; end.Cancel();
        }
        await Assert.ThrowsAsync<OperationCanceledException>(() => generator.GenerateAsync(AtEnd(), cancellationToken: end.Token));
        Assert.Equal(4, tokenized);
    }

    [Fact]
    public async Task Dispose_OwnedAndBorrowedEncodersAndServices()
    {
        using var session = TestAssets.Session();
        using var encoder = new OnnxTextEncoder(session, 3);
        var borrowed = new OnnxEmbeddingGenerator(Tokenizer(), Options(), encoder, PoolingMode.Mean, "contract", maximumBatchSize: 2);
        borrowed.Dispose(); borrowed.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => borrowed.GenerateAsync(["3"]));
        Assert.Throws<ObjectDisposedException>(() => borrowed.GetService(typeof(EmbeddingGeneratorMetadata)));
        Assert.Equal(new float[] {7,1,1,9,0,1}, encoder.Score(OnnxTextEncoderTests.Batch()).ToArray());
        var owned = new OnnxEmbeddingGenerator(Tokenizer(), Options(), encoder, PoolingMode.Mean, "contract", maximumBatchSize: 2, ownsEncoder: true);
        owned.Dispose(); owned.Dispose();
        Assert.Throws<ObjectDisposedException>(() => encoder.Score(OnnxTextEncoderTests.Batch()));
    }

    [Fact]
    public void Constructor_RejectsNullInvalidModeAndEmptyModel()
    {
        using var session = TestAssets.Session();
        using var encoder = new OnnxTextEncoder(session, 3);
        Assert.Throws<ArgumentNullException>(() => new OnnxEmbeddingGenerator(null!, Options(), encoder, PoolingMode.Mean, "contract"));
        Assert.Throws<ArgumentNullException>(() => new OnnxEmbeddingGenerator(Tokenizer(), null!, encoder, PoolingMode.Mean, "contract"));
        Assert.Throws<ArgumentNullException>(() => new OnnxEmbeddingGenerator(Tokenizer(), Options(), null!, PoolingMode.Mean, "contract"));
        Assert.Throws<ArgumentException>(() => new OnnxEmbeddingGenerator(Tokenizer(), Options(), encoder, PoolingMode.Mean, " "));
        Assert.Throws<ArgumentOutOfRangeException>(() => new OnnxEmbeddingGenerator(Tokenizer(), Options(), encoder, (PoolingMode)8, "contract"));
        foreach (int maximum in new[] {0, -1, int.MinValue})
            Assert.Throws<ArgumentOutOfRangeException>(() => new OnnxEmbeddingGenerator(Tokenizer(), Options(), encoder, PoolingMode.Mean, "contract", maximum));
    }

    [Fact]
    public void Generator_ExposesDirectCompositionWithoutPreparer()
    {
        using var session = TestAssets.Session();
        using var encoder = new OnnxTextEncoder(session, 3);
        var tokenizer = Tokenizer();
        var options = Options();
        using var generator = new OnnxEmbeddingGenerator(tokenizer, options, encoder, PoolingMode.Cls, "direct");
        Assert.Same(tokenizer, generator.Tokenizer);
        Assert.Same(options, generator.SequenceOptions);
        Assert.Same(encoder, generator.Encoder);
        Assert.Equal(32, generator.MaximumBatchSize);
        Assert.Equal(PoolingMode.Cls, generator.Pooling);
        Assert.Equal("direct", generator.ModelId);
        Assert.Null(typeof(OnnxEmbeddingGenerator).GetProperty("Preparer"));
    }

    [Fact]
    public async Task Generate_RowSlicesRemainOwnedAcrossCallsAndDisposal()
    {
        using var session = TestAssets.Session("required_types");
        using var encoder = new OnnxTextEncoder(session, 3, ownsSession: true);
        using var generator = new OnnxEmbeddingGenerator(Tokenizer(), Options(), encoder, PoolingMode.Mean, "rows",
            maximumBatchSize: 2, ownsEncoder: true);
        var results = await generator.GenerateAsync(["3", "6", "9", "12", "15"]);
        Assert.Equal(5, results.Count);
        var segments = results.Select(item =>
        {
            Assert.True(MemoryMarshal.TryGetArray(item.Vector, out var segment));
            Assert.Equal(3, segment.Count);
            Assert.Equal("rows", item.ModelId);
            return segment;
        }).ToArray();
        Assert.Same(segments[0].Array, segments[1].Array);
        Assert.Same(segments[2].Array, segments[3].Array);
        Assert.NotSame(segments[0].Array, segments[2].Array);
        Assert.NotSame(segments[2].Array, segments[4].Array);
        Assert.Equal(new[] {0,3,0,3,0}, segments.Select(segment => segment.Offset));
        Assert.Equal(new[] {6,6,6,6,3}, segments.Select(segment => segment.Array!.Length));
        ReadOnlyMemory<float> retainedFirst = results[0].Vector;
        ReadOnlyMemory<float> retainedLast = results[4].Vector;
        for (int row = 0; row < results.Count; row++) AssertMeanVector(results[row].Vector, row + 2);

        // The public MEAI setter replaces one row's view; it cannot overwrite the
        // shared batch storage or neighboring rows.
        results[0].Vector = new float[] {41,42,43};
        Assert.Equal(new float[] {41,42,43}, results[0].Vector.ToArray());
        AssertMeanVector(retainedFirst, 2);
        AssertMeanVector(results[1].Vector, 3);
        var later = await generator.GenerateAsync(["30", "60"]);
        AssertMeanVector(later[0].Vector, 11);
        AssertMeanVector(later[1].Vector, 21);
        Assert.True(MemoryMarshal.TryGetArray(later[0].Vector, out var laterSegment));
        Assert.NotSame(segments[0].Array, laterSegment.Array);
        Assert.NotSame(segments[4].Array, laterSegment.Array);
        generator.Dispose();
        Assert.True(OnnxTextEncoderTests.IsSessionDisposed(session));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(["3"]));
        results = null!;
        later = null!;
        segments = null!;
        GC.Collect();
        GC.WaitForPendingFinalizers();
        AssertMeanVector(retainedFirst, 2);
        AssertMeanVector(retainedLast, 6);
        Assert.True(MemoryMarshal.TryGetArray(retainedLast, out var retainedSegment));
        Assert.Equal(3, retainedSegment.Array!.Length); // one final partial batch, not the whole request
    }

    private static void AssertMeanVector(ReadOnlyMemory<float> actual, int meanId)
    {
        // Tiny required_types graph emits [ID, mask, type]; framed row is [1,content,2].
        double norm = Math.Sqrt(meanId * meanId + 1);
        Assert.Equal(3, actual.Length);
        Assert.InRange(Math.Abs(actual.Span[0] - meanId / norm), 0, 1e-6);
        Assert.InRange(Math.Abs(actual.Span[1] - 1 / norm), 0, 1e-6);
        Assert.Equal(0, actual.Span[2]);
    }
}
