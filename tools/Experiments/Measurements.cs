using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using CommunityToolkit.Embeddings.Onnx;
using Microsoft.ML.Tokenizers;
using Microsoft.Extensions.AI;
using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;

internal static class Measurements
{
    public static void Run(string assetRoot)
    {
        if (Environment.GetEnvironmentVariable("DOTNET_TieredCompilation") != "0")
            throw new InvalidOperationException("Set DOTNET_TieredCompilation=0 for this measurement process to avoid tier-transition noise.");
        var results = new List<object>();
        var startup = new List<object>();
        foreach (string model in new[] { "minilm", "granite" })
        {
            string folder = Path.Combine(assetRoot, model);
            int limit = model == "granite" ? 512 : 256;
            bool includeTypes = model == "minilm";
            long allocated = GC.GetAllocatedBytesForCurrentThread();
            long started = Stopwatch.GetTimestamp();
            Tokenizer tokenizer = model == "minilm"
                ? new BertUncasedTokenizer(Path.Combine(folder, "vocab.txt"))
                : new Granite30MEnglishTokenizer(Path.Combine(folder, "vocab.json"), Path.Combine(folder, "merges.txt"));
            double constructionUs = Stopwatch.GetElapsedTime(started).TotalMicroseconds;
            long constructionBytes = GC.GetAllocatedBytesForCurrentThread() - allocated;
            var sequence = new TokenSequenceOptions(limit, includeTypes ? 101 : 0, includeTypes ? 102 : 2, includeTypes ? 0 : 1);
            var firstPreparation = Measure(() => tokenizer.PrepareBatch(["An initial uncached sentence."], sequence, includeTokenTypeIds: includeTypes), 1);
            startup.Add(new { model, constructionUs, constructionBytes,
                firstPreparationUs = firstPreparation.Microseconds, firstPreparationBytes = firstPreparation.Bytes });
            foreach (int size in new[] { 1, 8, 32 })
            foreach (int length in new[] { 16, 128, limit })
            {
                string[] texts = Enumerable.Range(0, size)
                    .Select(i => string.Join(' ', Enumerable.Repeat("hello", length - 2 - i % 3))).ToArray();
                Func<object> baseline = () => PrepareBaseline(tokenizer, sequence, texts, includeTypes);
                Func<object> helper = () => tokenizer.PrepareBatch(texts, sequence, includeTokenTypeIds: includeTypes);
                TokenBatch prepared = tokenizer.PrepareBatch(texts, sequence, includeTokenTypeIds: includeTypes);
                TokenBatch simple = PrepareBaseline(tokenizer, sequence, texts, includeTypes);
                int elements = size * prepared.SequenceLength;
                if (!simple.InputIds.GetSpan([0, 0], elements).SequenceEqual(prepared.InputIds.GetSpan([0, 0], elements)) ||
                    !simple.AttentionMask.GetSpan([0, 0], elements).SequenceEqual(prepared.AttentionMask.GetSpan([0, 0], elements)) ||
                    (includeTypes && !simple.TokenTypeIds.GetSpan([0, 0], elements).SequenceEqual(prepared.TokenTypeIds.GetSpan([0, 0], elements))))
                    throw new InvalidOperationException("Preparation benchmark implementations differ.");
                results.Add(new { stage = "prepare", model, batch = size, sequence = prepared.SequenceLength, tokenTypeIds = includeTypes,
                    measurements = Compare(baseline, helper, 30) });

                var hidden = new float[size * prepared.SequenceLength * 384];
                for (int i = 0; i < hidden.Length; i++) hidden[i] = MathF.Sin(i * 0.1f) + 0.1f;
                Tensor<float> shaped = Tensor.Create(hidden, [size, prepared.SequenceLength, 384]);
                PoolingMode mode = model == "granite" ? PoolingMode.Cls : PoolingMode.Mean;
                Func<object> poolBaseline = () => PoolBaseline(hidden, prepared, 384, mode);
                Func<object> poolHelper = () => EmbeddingPooling.Pool(shaped, prepared.AttentionMask, mode);
                float[][] expected = PoolBaseline(hidden, prepared, 384, mode);
                Tensor<float> actual = EmbeddingPooling.Pool(shaped, prepared.AttentionMask, mode);
                for (int row = 0; row < size; row++)
                    for (int column = 0; column < 384; column++)
                        if (MathF.Abs(expected[row][column] - actual[row, column]) > 2e-6f)
                            throw new InvalidOperationException("Pooling benchmark implementations differ.");
                results.Add(new { stage = $"{mode.ToString().ToLowerInvariant()}-pool-normalize", model, batch = size, sequence = prepared.SequenceLength,
                    measurements = Compare(poolBaseline, poolHelper, 30) });
                if (length == limit)
                {
                    results.Add(new { stage = "pool-meai-ownership", model, batch = size, sequence = prepared.SequenceLength,
                        measurements = Compare(() => PoolAndCopyToMeai(shaped, prepared, mode),
                            () => PoolPrivateMeaiBuffer(shaped, prepared, mode), 30) });
                    results.Add(new { stage = "pool-meai-jagged-storage", model, batch = size, sequence = prepared.SequenceLength,
                        measurements = Compare(() => PoolJaggedMeai(hidden, prepared, mode),
                            () => PoolPrivateMeaiBuffer(shaped, prepared, mode), 30) });
                    var reusable = new float[size * 384];
                    results.Add(new { stage = "pool-destination-reuse", model, batch = size, sequence = prepared.SequenceLength,
                        measurements = Compare(poolHelper, () =>
                        {
                            EmbeddingPooling.PoolInto(shaped, prepared.AttentionMask, new TensorSpan<float>(reusable, [size, 384]), mode);
                            return reusable;
                        }, 30) });
                }
            }
            using var options = new SessionOptions { IntraOpNumThreads = 2 };
            using var encoder = OnnxTextEncoder.Load(Path.Combine(folder, "model.onnx"), options,
                outputName: model == "granite" ? "logits" : "last_hidden_state");
            PoolingMode endToEndMode = model == "granite" ? PoolingMode.Cls : PoolingMode.Mean;
            using var generator = new OnnxEmbeddingGenerator(tokenizer, sequence, encoder, endToEndMode, model);
            string[] inputs = Enumerable.Range(0, 8)
                .Select(i => string.Join(' ', Enumerable.Repeat("hello", 126 - i % 3))).ToArray();
            Func<GeneratedEmbeddings<Embedding<float>>> copiedPipeline = () =>
            {
                TokenBatch batch = tokenizer.PrepareBatch(inputs, sequence, includeTokenTypeIds: encoder.RequiresTokenTypeIds);
                return PoolAndCopyToMeai(encoder.Score(batch), batch, endToEndMode);
            };
            Func<GeneratedEmbeddings<Embedding<float>>> providerPipeline = () => generator.GenerateAsync(inputs).GetAwaiter().GetResult();
            var expectedEndToEnd = copiedPipeline();
            var actualEndToEnd = providerPipeline();
            for (int row = 0; row < inputs.Length; row++)
                if (!expectedEndToEnd[row].Vector.Span.SequenceEqual(actualEndToEnd[row].Vector.Span))
                    throw new InvalidOperationException("End-to-end pipeline vectors differ.");
            results.Add(new { stage = "prepare-score-pool-meai", model, batch = 8, sequence = 128,
                measurements = Compare(copiedPipeline, providerPipeline, 3) });
        }
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            runtime = RuntimeInformation.FrameworkDescription,
            os = RuntimeInformation.OSDescription,
            architecture = RuntimeInformation.ProcessArchitecture.ToString(),
            processors = Environment.ProcessorCount,
            tieredCompilation = false,
            implementation = "shaped-tensor-v1",
            note = "Median of 7 alternating-order rounds, 30 calls (3 for end-to-end), 10 warmups. Current-thread managed allocations, cached repeated texts. Startup/input construction excluded. Only prepare-score-pool-meai includes native inference, not native-memory accounting; both end-to-end paths retain the same one native-output copy. End-to-end baseline copies pooled rows; helper is actual provider with private final buffer.",
            startup,
            results
        }, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static GeneratedEmbeddings<Embedding<float>> PoolAndCopyToMeai(Tensor<float> hidden, TokenBatch batch, PoolingMode mode)
    {
        Tensor<float> pooled = EmbeddingPooling.Pool(hidden, batch.AttentionMask, mode);
        var result = new GeneratedEmbeddings<Embedding<float>>();
        for (int row = 0; row < batch.BatchSize; row++)
            result.Add(new(pooled.GetSpan([row, 0], 384).ToArray()));
        return result;
    }

    private static GeneratedEmbeddings<Embedding<float>> PoolJaggedMeai(float[] hidden, TokenBatch batch, PoolingMode mode)
    {
        float[][] pooled = PoolBaseline(hidden, batch, 384, mode);
        var result = new GeneratedEmbeddings<Embedding<float>>();
        foreach (float[] row in pooled) result.Add(new(row));
        return result;
    }

    private static GeneratedEmbeddings<Embedding<float>> PoolPrivateMeaiBuffer(Tensor<float> hidden, TokenBatch batch, PoolingMode mode)
    {
        var buffer = new float[batch.BatchSize * 384];
        EmbeddingPooling.PoolInto(hidden, batch.AttentionMask, new TensorSpan<float>(buffer, [batch.BatchSize, 384]), mode);
        var result = new GeneratedEmbeddings<Embedding<float>>();
        for (int row = 0; row < batch.BatchSize; row++)
            result.Add(new(buffer.AsMemory(row * 384, 384)));
        return result;
    }

    private static object Compare(Func<object> baseline, Func<object> helper, int iterations)
    {
        var firstBaseline = Measure(baseline, 1);
        var firstHelper = Measure(helper, 1);
        for (int i = 0; i < 10; i++) { GC.KeepAlive(baseline()); GC.KeepAlive(helper()); }
        var baselines = new List<(double Microseconds, long Bytes)>();
        var helpers = new List<(double Microseconds, long Bytes)>();
        for (int i = 0; i < 7; i++)
        {
            if (i % 2 == 0) { baselines.Add(Measure(baseline, iterations)); helpers.Add(Measure(helper, iterations)); }
            else { helpers.Add(Measure(helper, iterations)); baselines.Add(Measure(baseline, iterations)); }
        }
        return new
        {
            firstMeasuredBaselineUs = firstBaseline.Microseconds,
            firstMeasuredHelperUs = firstHelper.Microseconds,
            baselineUs = baselines.OrderBy(x => x.Microseconds).ElementAt(3).Microseconds,
            helperUs = helpers.OrderBy(x => x.Microseconds).ElementAt(3).Microseconds,
            baselineBytes = baselines.OrderBy(x => x.Bytes).ElementAt(3).Bytes,
            helperBytes = helpers.OrderBy(x => x.Bytes).ElementAt(3).Bytes
        };
    }

    private static (double Microseconds, long Bytes) Measure(Func<object> operation, int iterations)
    {
        long allocated = GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        for (int i = 0; i < iterations; i++) GC.KeepAlive(operation());
        double elapsed = Stopwatch.GetElapsedTime(start).TotalMicroseconds / iterations;
        return (elapsed, (GC.GetAllocatedBytesForCurrentThread() - allocated) / iterations);
    }

    private static TokenBatch PrepareBaseline(Tokenizer tokenizer, TokenSequenceOptions sequence, string[] texts, bool includeTypes)
    {
        long[][] sequences = texts.Select(text => new[] { sequence.BeginningTokenId }
            .Concat(tokenizer.EncodeToIds(text).Take(sequence.MaximumSequenceLength - 2)).Append(sequence.EndTokenId)
            .Select(id => (long)id).ToArray()).ToArray();
        int length = sequences.Max(ids => ids.Length);
        long[] ids = sequences.SelectMany(row => row.Concat(Enumerable.Repeat((long)sequence.PaddingTokenId, length - row.Length))).ToArray();
        long[] mask = sequences.SelectMany(row => Enumerable.Repeat(1L, row.Length).Concat(Enumerable.Repeat(0L, length - row.Length))).ToArray();
        return new TokenBatch(texts.Length, length, ids, mask, includeTypes ? new long[ids.Length] : ReadOnlySpan<long>.Empty);
    }

    private static float[][] PoolBaseline(float[] hidden, TokenBatch batch, int dimensions, PoolingMode mode)
    {
        var result = new float[batch.BatchSize][];
        var mask = batch.AttentionMask.GetSpan([0, 0], batch.BatchSize * batch.SequenceLength);
        foreach (float value in hidden)
            if (!float.IsFinite(value)) throw new InvalidOperationException("Nonfinite output.");
        for (int row = 0; row < batch.BatchSize; row++)
        {
            var vector = new float[dimensions];
            int tokens = 0;
            if (mode == PoolingMode.Cls)
            {
                if (mask[row * batch.SequenceLength] != 1) throw new InvalidOperationException("Masked CLS token.");
                tokens = 1;
                for (int col = 0; col < dimensions; col++) vector[col] = hidden[row * batch.SequenceLength * dimensions + col];
            }
            else
            {
                for (int token = 0; token < batch.SequenceLength; token++)
                {
                    int position = row * batch.SequenceLength + token;
                    if (mask[position] == 0) continue;
                    tokens++;
                    for (int col = 0; col < dimensions; col++) vector[col] += hidden[position * dimensions + col];
                }
            }
            if (tokens == 0) throw new InvalidOperationException("All-padding row.");
            float maximum = 0;
            for (int col = 0; col < dimensions; col++)
            {
                vector[col] /= tokens;
                if (!float.IsFinite(vector[col])) throw new InvalidOperationException("Pooling overflow.");
                maximum = Math.Max(maximum, Math.Abs(vector[col]));
            }
            if (maximum == 0) throw new InvalidOperationException("Zero vector.");
            float squareSum = 0;
            for (int col = 0; col < dimensions; col++)
            {
                vector[col] /= maximum;
                squareSum += vector[col] * vector[col];
            }
            float norm = MathF.Sqrt(squareSum);
            for (int col = 0; col < dimensions; col++) vector[col] /= norm;
            result[row] = vector;
        }
        return result;
    }
}
