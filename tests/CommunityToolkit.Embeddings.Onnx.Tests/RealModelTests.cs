using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using CommunityToolkit.Embeddings.Onnx;
using Microsoft.Extensions.AI;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Tokenizers;
using Xunit.Abstractions;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

/// <summary>
/// Actual publisher models versus independent offline HuggingFace tokenization
/// and PyTorch float32 goldens. Synthetic graphs are NOT used in this class.
/// </summary>
[Trait("Category", "RealModels")]
public sealed class RealModelTests(ITestOutputHelper output)
{
    private const double AbsoluteTolerance = 2e-5;
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true, WriteIndented = true };
    private static readonly Dictionary<string, (string Revision, string Repository, int Length)> Pins = new()
    {
        ["minilm"] = ("1110a243fdf4706b3f48f1d95db1a4f5529b4d41", AllMiniLmL6V2EmbeddingGenerator.SupportedModelId, 256),
        ["e5"] = ("ffb93f3bd4047442299a41ebb6fa998a38507c52", E5SmallV2EmbeddingGenerator.SupportedModelId, 512),
        ["granite"] = ("9b5b096411652ec1189c68fcfb90d0a82c5b45af", GraniteEmbedding30MEnglishGenerator.SupportedModelId, 512)
    };

    [Theory]
    [InlineData("minilm")]
    [InlineData("e5")]
    [InlineData("granite")]
    public async Task PinnedModels_MatchIndependentTokensAndVectors(string key)
    {
        var reference = ReadReference(key);
        Record(key, "parity", new { passed = false });
        using var session = Session(key);
        using var encoder = new OnnxTextEncoder(session, 384, key == "granite" ? "logits" : "last_hidden_state");
        using var provider = Provider(key, session);
        var preparer = new TextBatchPreparer(Tokenizer(key), SequenceOptions(key), 6);
        AssertAllGoldenTokens(key, reference, preparer);
        double stageMax = 0, providerMax = 0, normError = 0, minimumCosine = 1;
        int count = 0;
        foreach (var golden in reference.Batches)
        {
            string[] texts = golden.Items.Select(item => Formatted(key, item)).ToArray();
            var tokens = preparer.Prepare(texts);
            Assert.Equal(golden.Items.Length, tokens.BatchSize);
            Assert.Equal(golden.InputIds[0].Length, tokens.SequenceLength);
            Assert.Equal(golden.InputIds.SelectMany(x => x), tokens.InputIds.ToArray());
            Assert.Equal(golden.AttentionMask.SelectMany(x => x), tokens.AttentionMask.ToArray());
            Assert.Equal(golden.TokenTypeIds.SelectMany(x => x), tokens.TokenTypeIds.ToArray());
            Assert.True(tokens.HasTokenTypeIds);
            float[] hidden = encoder.Score(tokens);
            Assert.Equal(tokens.BatchSize * tokens.SequenceLength * 384, hidden.Length);
            float[][] composed = EmbeddingPooling.Pool(hidden, tokens, 384, key == "granite" ? PoolingMode.Cls : PoolingMode.Mean);
            var task = Generate(provider, key, golden.Items);
            Assert.True(task.IsCompletedSuccessfully);
            var concrete = await task;
            Assert.Equal(golden.Items.Length, concrete.Count);
            for (int row = 0; row < golden.Items.Length; row++)
            {
                var stage = Compare(golden.Vectors[row], composed[row], $"{key} stage batch {count} row {row}");
                var actual = Compare(golden.Vectors[row], concrete[row].Vector.ToArray(), $"{key} provider batch {count} row {row}");
                stageMax = Math.Max(stageMax, stage.MaxAbsolute);
                providerMax = Math.Max(providerMax, actual.MaxAbsolute);
                minimumCosine = Math.Min(minimumCosine, Math.Min(stage.Cosine, actual.Cosine));
                normError = Math.Max(normError, Math.Max(stage.NormError, actual.NormError));
                Assert.Equal(Pins[key].Repository, concrete[row].ModelId);
            }
            count += golden.Items.Length;
        }
        Assert.Equal(key == "e5" ? 67 : 36, count);
        Assert.Equal(2, provider.Preparer.MaximumBatchSize); // goldens are batches of 6, provider must rechunk
        output.WriteLine($"{key}: {count} rows; stage max abs={stageMax:G9}, provider max abs={providerMax:G9}, min cosine={minimumCosine:G12}, max norm error={normError:G9}");
        Record(key, "parity", new { passed = true, count, stageMaxAbsoluteError = stageMax, providerMaxAbsoluteError = providerMax, minimumCosine, maximumNormError = normError, providerMaximumBatchSize = 2 });
    }

    [Theory]
    [InlineData("minilm")]
    [InlineData("e5")]
    [InlineData("granite")]
    public async Task PinnedModels_SingleAndConcurrentInferenceMatchMixedBatches(string key)
    {
        var reference = ReadReference(key);
        Record(key, "batchAndConcurrency", new { passed = false });
        using var session = Session(key);
        using var provider = Provider(key, session);
        double maximum = 0;
        // Every row is compared alone, including long/truncated inputs and both E5 roles.
        foreach (var batch in reference.Batches)
        {
            var mixed = await Generate(provider, key, batch.Items);
            for (int row = 0; row < batch.Items.Length; row++)
            {
                var single = Assert.Single(await Generate(provider, key, [batch.Items[row]]));
                maximum = Math.Max(maximum, Compare(mixed[row].Vector.ToArray(), single.Vector.ToArray(), $"{key} alone vs mixed").MaxAbsolute);
                Compare(batch.Vectors[row], single.Vector.ToArray(), $"{key} alone vs PyTorch");
            }
        }
        var shortBatch = reference.Batches[1]; // spaces, Unicode, CJK, emoji and astral letters/numbers
        using var start = new Barrier(3);
        var concurrent = await Task.WhenAll(Enumerable.Range(0, 3)
            .Select(_ => Task.Factory.StartNew(() =>
            {
                start.SignalAndWait();
                return Generate(provider, key, shortBatch.Items);
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default).Unwrap()));
        foreach (var results in concurrent)
        {
            Assert.Equal(shortBatch.Items.Length, results.Count);
            for (int row = 0; row < results.Count; row++)
                Compare(shortBatch.Vectors[row], results[row].Vector.ToArray(), $"{key} concurrent row {row}");
        }
        // Mutating one returned owned buffer must not affect a neighboring result or a later call.
        var first = concurrent[0][0].Vector;
        Assert.True(System.Runtime.InteropServices.MemoryMarshal.TryGetArray(first, out var segment));
        float[] neighbor = concurrent[0][1].Vector.ToArray();
        segment.Array![segment.Offset] = 1234;
        Assert.Equal(neighbor, concurrent[0][1].Vector.ToArray());
        Compare(shortBatch.Vectors[0], concurrent[1][0].Vector.ToArray(), $"{key} independent concurrent buffer");
        Compare(shortBatch.Vectors[0], Assert.Single(await Generate(provider, key, [shortBatch.Items[0]])).Vector.ToArray(), $"{key} independent later buffer");
        output.WriteLine($"{key}: same text alone vs mixed max absolute error={maximum:G9}; three concurrent callers passed");
        Record(key, "batchAndConcurrency", new { passed = true, aloneVersusMixedMaximumAbsoluteError = maximum, concurrentCallers = 3 });
    }

    [Theory]
    [InlineData(E5Purpose.Query)]
    [InlineData(E5Purpose.Document)]
    public async Task E5_DefaultSingleRoleViaMeaiAndTypedMixedRolesMatchGoldens(E5Purpose purpose)
    {
        var reference = ReadReference("e5");
        using var session = Session("e5");
        using var concrete = new E5SmallV2EmbeddingGenerator(AssetDirectory("e5"), purpose, session, maximumBatchSize: 2);
        IEmbeddingGenerator<string, Embedding<float>> meai = concrete;
        var cases = reference.Batches.SelectMany(batch => batch.Items.Select((item, index) => (Item: item, Vector: batch.Vectors[index])))
            .Where(pair => pair.Item.Purpose == purpose.ToString()).ToArray();
        Assert.Equal(purpose == E5Purpose.Query ? 35 : 32, cases.Length);
        var generated = await meai.GenerateAsync(cases.Select(pair => pair.Item.Text));
        Assert.Equal(cases.Length, generated.Count);
        Assert.Equal(purpose, concrete.Purpose);
        for (int row = 0; row < cases.Length; row++)
            Compare(cases[row].Vector, generated[row].Vector.ToArray(), $"E5 default {purpose} row {row}");
        var mixed = reference.Batches[1];
        Assert.Contains(mixed.Items, item => item.Purpose == "Query");
        Assert.Contains(mixed.Items, item => item.Purpose == "Document");
        var typed = await concrete.GenerateAsync(mixed.Items.Select(item => new E5Input(item.Text, Enum.Parse<E5Purpose>(item.Purpose!))));
        Assert.Equal(mixed.Items.Length, typed.Count);
        for (int row = 0; row < typed.Count; row++)
            Compare(mixed.Vectors[row], typed[row].Vector.ToArray(), $"E5 explicit mixed ignores default {purpose} row {row}");
        Assert.Empty(await concrete.GenerateAsync(Array.Empty<E5Input>()));
        await Assert.ThrowsAsync<ArgumentNullException>(() => concrete.GenerateAsync((IEnumerable<E5Input>)null!));
        await Assert.ThrowsAsync<ArgumentNullException>(() => concrete.GenerateAsync([new E5Input(null!, purpose)]));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => concrete.GenerateAsync([new E5Input("x", (E5Purpose)99)]));
        Record("e5", $"default{purpose}", new { passed = true, count = generated.Count, mixedCount = typed.Count });
    }

    [Theory]
    [InlineData("minilm")]
    [InlineData("e5")]
    [InlineData("granite")]
    public async Task PinnedModels_ConcreteValidationOwnershipAndRetrievalRanking(string key)
    {
        var reference = ReadReference(key);
        using var options = TestAssets.Options();
        using var session = new InferenceSession(AssetFile(key, "model.onnx"), options);
        var provider = Provider(key, session);
        Assert.Throws<ArgumentOutOfRangeException>(() => Provider(key, session, maximumBatchSize: 0));
        if (key == "e5")
            Assert.Throws<ArgumentOutOfRangeException>(() => new E5SmallV2EmbeddingGenerator(AssetDirectory(key), (E5Purpose)99, session));
        Assert.Equal(384, provider.Encoder.Dimensions);
        Assert.Equal(Pins[key].Length, provider.Preparer.MaximumSequenceLength);
        Assert.Equal(key == "granite" ? PoolingMode.Cls : PoolingMode.Mean, provider.Pooling);
        var metadata = Assert.IsType<EmbeddingGeneratorMetadata>(provider.GetService(typeof(EmbeddingGeneratorMetadata)));
        Assert.Equal(384, metadata.DefaultModelDimensions);
        Assert.Equal(Pins[key].Repository, metadata.DefaultModelId);
        Assert.Null(provider.GetService(typeof(EmbeddingGeneratorMetadata), "nondefault"));
        Assert.Same(provider, provider.GetService(typeof(IEmbeddingGenerator<string, Embedding<float>>)));
        Assert.Empty(await provider.GenerateAsync(Array.Empty<string>()));
        await Assert.ThrowsAsync<ArgumentNullException>(() => provider.GenerateAsync((IEnumerable<string>)null!));
        await Assert.ThrowsAsync<ArgumentNullException>(() => provider.GenerateAsync(new string[] {null!}));
        await Assert.ThrowsAsync<ArgumentException>(() => provider.GenerateAsync(["text"], new() { Dimensions = 383 }));
        await Assert.ThrowsAsync<ArgumentException>(() => provider.GenerateAsync(["text"], new() { ModelId = "wrong" }));
        var accepted = await provider.GenerateAsync(["hello"], new() { Dimensions = 384, ModelId = Pins[key].Repository });
        Assert.Equal(384, Assert.Single(accepted).Vector.Length);
        if (key == "granite")
        {
            Assert.Equal(3, session.OutputMetadata["logits"].Dimensions.Length);
            Assert.Equal(2, session.OutputMetadata["894"].Dimensions.Length);
            Assert.Throws<ArgumentException>(() => new OnnxTextEncoder(session, outputName: "894"));
        }
        var corpus = reference.Batches.SelectMany(batch => batch.Items.Select((item, index) => (Item: item, Vector: batch.Vectors[index]))).ToArray();
        string[] retrieval = ["A dog is playing in the park.", "A puppy plays outdoors.", "Database indexes accelerate queries."];
        var selected = retrieval.Select((text, index) => corpus.Single(pair => pair.Item.Text == text &&
            (key != "e5" || pair.Item.Purpose == (index == 0 ? "Query" : "Document")))).ToArray();
        var actual = await Generate(provider, key, selected.Select(pair => pair.Item).ToArray());
        Assert.Equal(3, actual.Count);
        for (int i = 0; i < 3; i++) Compare(selected[i].Vector, actual[i].Vector.ToArray(), $"{key} retrieval vector {i}");
        double related = Cosine(actual[0].Vector.ToArray(), actual[1].Vector.ToArray());
        double unrelated = Cosine(actual[0].Vector.ToArray(), actual[2].Vector.ToArray());
        Assert.True(related > unrelated, $"{key}: dog/puppy {related} must rank above dog/database {unrelated}.");
        Assert.True(Cosine(selected[0].Vector, selected[1].Vector) > Cosine(selected[0].Vector, selected[2].Vector));
        provider.Dispose(); provider.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => provider.GenerateAsync(["hello"]));
        Assert.Throws<ObjectDisposedException>(() => provider.GetService(typeof(EmbeddingGeneratorMetadata)));
        // The concrete provider owns its wrapper, but borrows the supplied session by default.
        using var survivor = Provider(key, session);
        Compare(selected[0].Vector, Assert.Single(await Generate(survivor, key, [selected[0].Item])).Vector.ToArray(), $"{key} borrowed session survives");
        Assert.False(options.IsClosed);
        using var loaded = LoadedProvider(key, options);
        Assert.Equal(384, Assert.Single(await loaded.GenerateAsync(["hello"])).Vector.Length);
        loaded.Dispose();
        Assert.False(options.IsClosed);
        using var ownedSession = Session(key);
        var owning = Provider(key, ownedSession, ownsSession: true);
        Assert.False(OnnxTextEncoderTests.IsSessionDisposed(ownedSession));
        owning.Dispose(); owning.Dispose();
        Assert.True(OnnxTextEncoderTests.IsSessionDisposed(ownedSession));
        Record(key, "retrievalAndContracts", new { passed = true, relatedCosine = related, unrelatedCosine = unrelated });
    }

    [Theory]
    [InlineData("minilm")]
    [InlineData("e5")]
    [InlineData("granite")]
    public void PinnedModels_StandardTokenizerMatchesIndependentUnicodeSpecialAndWhitespaceEvidence(string key)
    {
        string path = Path.Combine(TestAssets.Root, ".assets", "tokenizer-contract-reference.json");
        Assert.True(File.Exists(path), "Generate separate offline tokenizer evidence with dev/tokenizer_reference.py; never replace model reference.json.");
        using var reference = JsonDocument.Parse(File.ReadAllText(path));
        Assert.Equal("4.57.6", reference.RootElement.GetProperty("packages").GetProperty("transformers").GetString());
        Assert.Equal("0.22.2", reference.RootElement.GetProperty("packages").GetProperty("tokenizers").GetString());
        var rows = reference.RootElement.GetProperty("models").GetProperty(key);
        Assert.Equal(7, rows.GetArrayLength());
        Tokenizer tokenizer = Tokenizer(key);
        foreach (var row in rows.EnumerateArray())
        {
            string text = row.GetProperty("text").GetString()!;
            int[] ids = row.GetProperty("ids").EnumerateArray().Select(value => value.GetInt32()).ToArray();
            string[] values = row.GetProperty("tokens").EnumerateArray().Select(value => value.GetString()!).ToArray();
            Assert.Equal(ids, tokenizer.EncodeToIds(text));
            Assert.Equal(ids, tokenizer.EncodeToIds(text.AsSpan()));
            Assert.Equal(ids, tokenizer.EncodeToTokens(text, out _).Select(token => token.Id));
            Assert.Equal(values, tokenizer.EncodeToTokens(text.AsSpan(), out _).Select(token => token.Value));
            Assert.Equal(ids.Length, tokenizer.CountTokens(text));
            Assert.Equal(ids.Length, tokenizer.CountTokens(text.AsSpan()));
            if (key == "granite")
            {
                string expected = row.GetProperty("decoded").GetString()!;
                Assert.Equal(expected, tokenizer.Decode(ids));
                char[] destination = new char[expected.Length];
                Assert.Equal(System.Buffers.OperationStatus.Done, tokenizer.Decode(ids, destination, out int consumed, out int written));
                Assert.Equal(ids.Length, consumed);
                Assert.Equal(expected.Length, written);
                Assert.Equal(expected, new string(destination));
            }
        }
        Record(key, "tokenizerContract", new
        {
            passed = true, count = rows.GetArrayLength(), transformers = "4.57.6", tokenizers = "0.22.2",
            stringSpanIdsTokensAndCount = true, independentGraniteDecode = key == "granite"
        });
    }

    private static string AssetDirectory(string key) => Path.Combine(TestAssets.Root, ".assets", key);
    private static void AssertAllGoldenTokens(string key, GoldenReference reference, TextBatchPreparer preparer)
    {
        var failures = new List<string>();
        int position = 0;
        foreach (var golden in reference.Batches)
        {
            var batch = preparer.Prepare(golden.Items.Select(item => Formatted(key, item)).ToArray());
            for (int row = 0; row < golden.Items.Length; row++, position++)
            {
                long[] actual = batch.InputIds.Slice(row * batch.SequenceLength, batch.SequenceLength).ToArray();
                if (golden.InputIds[row].SequenceEqual(actual)) continue;
                int mismatch = Enumerable.Range(0, Math.Min(actual.Length, golden.InputIds[row].Length))
                    .FirstOrDefault(i => actual[i] != golden.InputIds[row][i], -1);
                string preview = golden.Items[row].Text.Length > 100 ? golden.Items[row].Text[..100] + "..." : golden.Items[row].Text;
                failures.Add($"{key} row {position} ({golden.Items[row].Purpose ?? "none"}), text={JsonSerializer.Serialize(preview)}, " +
                    $"first ID difference at {mismatch}, expected prefix=[{string.Join(',', golden.InputIds[row].Take(25))}], " +
                    $"actual prefix=[{string.Join(',', actual.Take(25))}], expected length={golden.InputIds[row].Length}, actual length={actual.Length}");
            }
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }
    private static string AssetFile(string key, string file)
    {
        string path = Path.Combine(AssetDirectory(key), file);
        Assert.True(File.Exists(path), $"Missing real-model asset: {path}. Provision publisher {Pins[key].Repository} at revision {Pins[key].Revision}; generate independent offline HF/PyTorch reference.json with dev/reference.py. RealModels tests never skip or download.");
        return path;
    }
    private static GoldenReference ReadReference(string key)
    {
        AssetFile(key, "model.onnx");
        AssetFile(key, key == "granite" ? "vocab.json" : "vocab.txt");
        if (key == "granite") AssetFile(key, "merges.txt");
        var reference = JsonSerializer.Deserialize<GoldenReference>(File.ReadAllText(AssetFile(key, "reference.json")), JsonOptions)!;
        Assert.Equal(key, reference.Model.Key);
        Assert.Equal(Pins[key].Revision, reference.Model.Revision);
        Assert.Equal(Pins[key].Repository, reference.Model.Repository);
        Assert.Equal(Pins[key].Length, reference.Model.MaximumLength);
        Assert.Equal(key == "granite" ? "cls" : "mean", reference.Model.Pooling);
        Assert.NotEmpty(reference.Batches);
        return reference;
    }
    private static InferenceSession Session(string key)
    {
        using var options = TestAssets.Options();
        return new InferenceSession(AssetFile(key, "model.onnx"), options);
    }
    private static Tokenizer Tokenizer(string key) => key == "granite"
        ? new Granite30MEnglishTokenizer(AssetFile(key, "vocab.json"), AssetFile(key, "merges.txt"))
        : new BertUncasedTokenizer(AssetFile(key, "vocab.txt"));
    private static TokenSequenceOptions SequenceOptions(string key) => key == "granite"
        ? new(Pins[key].Length, 0, 2, 1) : new(Pins[key].Length, 101, 102, 0);
    private static OnnxEmbeddingGenerator Provider(string key, InferenceSession session, bool ownsSession = false, int maximumBatchSize = 2) => key switch
    {
        "minilm" => new AllMiniLmL6V2EmbeddingGenerator(AssetDirectory(key), session, ownsSession, maximumBatchSize),
        "e5" => new E5SmallV2EmbeddingGenerator(AssetDirectory(key), E5Purpose.Query, session, ownsSession, maximumBatchSize),
        "granite" => new GraniteEmbedding30MEnglishGenerator(AssetDirectory(key), session, ownsSession, maximumBatchSize),
        _ => throw new ArgumentOutOfRangeException(nameof(key))
    };
    private static OnnxEmbeddingGenerator LoadedProvider(string key, SessionOptions options) => key switch
    {
        "minilm" => new AllMiniLmL6V2EmbeddingGenerator(AssetDirectory(key), options, maximumBatchSize: 2),
        "e5" => new E5SmallV2EmbeddingGenerator(AssetDirectory(key), E5Purpose.Query, options, maximumBatchSize: 2),
        "granite" => new GraniteEmbedding30MEnglishGenerator(AssetDirectory(key), options, maximumBatchSize: 2),
        _ => throw new ArgumentOutOfRangeException(nameof(key))
    };
    private static string Formatted(string key, GoldenItem item) => key == "e5"
        ? E5Text.Format(item.Text, Enum.Parse<E5Purpose>(item.Purpose!)) : item.Text;
    private static Task<GeneratedEmbeddings<Embedding<float>>> Generate(OnnxEmbeddingGenerator provider, string key, GoldenItem[] items)
        => key == "e5" ? ((E5SmallV2EmbeddingGenerator)provider).GenerateAsync(items.Select(item => new E5Input(item.Text, Enum.Parse<E5Purpose>(item.Purpose!))))
            : provider.GenerateAsync(items.Select(item => item.Text));

    private static (double MaxAbsolute, double Cosine, double NormError) Compare(float[] expected, float[] actual, string context)
    {
        Assert.Equal(384, expected.Length);
        Assert.Equal(384, actual.Length);
        Assert.All(actual, value => Assert.True(float.IsFinite(value), $"{context}: nonfinite vector"));
        double max = expected.Zip(actual, (a,b) => Math.Abs((double)a - b)).Max();
        double cosine = Cosine(expected, actual);
        double normError = Math.Abs(Math.Sqrt(actual.Sum(x => (double)x * x)) - 1);
        Assert.True(max <= AbsoluteTolerance, $"{context}: max absolute error {max:G12} exceeds {AbsoluteTolerance}.");
        Assert.True(cosine >= .99999, $"{context}: cosine {cosine:G12} below 0.99999.");
        Assert.True(normError <= 2e-6, $"{context}: norm error {normError:G12} exceeds 2e-6.");
        return (max, cosine, normError);
    }
    private static double Cosine(float[] a, float[] b) => a.Zip(b, (x,y) => (double)x * y).Sum() /
        Math.Sqrt(a.Sum(x => (double)x * x) * b.Sum(x => (double)x * x));
    private static void Record(string key, string section, object values)
    {
        string path = Path.Combine(TestAssets.Root, ".assets", "dotnet-validation.json");
        var root = File.Exists(path) ? JsonNode.Parse(File.ReadAllText(path))!.AsObject() : new JsonObject();
        root["tolerances"] = JsonSerializer.SerializeToNode(new { absolute = AbsoluteTolerance, cosineMinimum = .99999, normError = 2e-6 });
        root[key] ??= new JsonObject();
        root[key]!["revision"] = Pins[key].Revision;
        root[key]![section] = JsonSerializer.SerializeToNode(values);
        File.WriteAllText(path, root.ToJsonString(JsonOptions));
    }
    public sealed class GoldenReference
    {
        public GoldenModel Model { get; set; } = new();
        public GoldenBatch[] Batches { get; set; } = [];
    }
    public sealed class GoldenModel
    {
        public string Key { get; set; } = "";
        public string Repository { get; set; } = "";
        public string Revision { get; set; } = "";
        [JsonPropertyName("max_length")] public int MaximumLength { get; set; }
        public string Pooling { get; set; } = "";
    }
    public sealed class GoldenBatch
    {
        public GoldenItem[] Items { get; set; } = [];
        public long[][] InputIds { get; set; } = [];
        public long[][] AttentionMask { get; set; } = [];
        public long[][] TokenTypeIds { get; set; } = [];
        public float[][] Vectors { get; set; } = [];
    }
    public sealed class GoldenItem
    {
        public string Text { get; set; } = "";
        public string? Purpose { get; set; }
    }
}
