using CommunityToolkit.Embeddings.Onnx;
using Microsoft.Extensions.AI;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

public sealed class OnnxEmbeddingGeneratorTests
{
    private static TextBatchPreparer Preparer(Func<string, IReadOnlyList<int>>? encode = null)
        => new(new TextTokenizer(encode ?? (text => [int.Parse(text)]), 1, 2, 0), 8, 2);

    [Fact]
    public async Task Generate_CompletesSynchronouslyChunksAndPreservesOrder()
    {
        using var session = TestAssets.Session("required_types");
        using var encoder = new OnnxTextEncoder(session, 3);
        var calls = new List<string>();
        int callerThread = Environment.CurrentManagedThreadId;
        using var generator = new OnnxEmbeddingGenerator(Preparer(text =>
        {
            Assert.Equal(callerThread, Environment.CurrentManagedThreadId);
            calls.Add(text);
            return [int.Parse(text)];
        }), encoder, PoolingMode.Mean, "contract");
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
        using var generator = new OnnxEmbeddingGenerator(Preparer(), encoder, PoolingMode.Mean, "contract");
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
        using var generator = new OnnxEmbeddingGenerator(Preparer(_ => { tokenized++; return [3]; }), encoder, PoolingMode.Mean, "contract");
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
        var borrowed = new OnnxEmbeddingGenerator(Preparer(), encoder, PoolingMode.Mean, "contract");
        borrowed.Dispose(); borrowed.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => borrowed.GenerateAsync(["3"]));
        Assert.Throws<ObjectDisposedException>(() => borrowed.GetService(typeof(EmbeddingGeneratorMetadata)));
        Assert.Equal(new float[] {7,1,1,9,0,1}, encoder.Score(OnnxTextEncoderTests.Batch()));
        var owned = new OnnxEmbeddingGenerator(Preparer(), encoder, PoolingMode.Mean, "contract", ownsEncoder: true);
        owned.Dispose(); owned.Dispose();
        Assert.Throws<ObjectDisposedException>(() => encoder.Score(OnnxTextEncoderTests.Batch()));
    }

    [Fact]
    public void Constructor_RejectsNullInvalidModeAndEmptyModel()
    {
        using var session = TestAssets.Session();
        using var encoder = new OnnxTextEncoder(session, 3);
        Assert.Throws<ArgumentNullException>(() => new OnnxEmbeddingGenerator(null!, encoder, PoolingMode.Mean, "contract"));
        Assert.Throws<ArgumentNullException>(() => new OnnxEmbeddingGenerator(Preparer(), null!, PoolingMode.Mean, "contract"));
        Assert.Throws<ArgumentException>(() => new OnnxEmbeddingGenerator(Preparer(), encoder, PoolingMode.Mean, " "));
        Assert.Throws<ArgumentOutOfRangeException>(() => new OnnxEmbeddingGenerator(Preparer(), encoder, (PoolingMode)8, "contract"));
    }
}
