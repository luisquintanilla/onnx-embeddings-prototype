using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using CommunityToolkit.Embeddings.Onnx;
using Microsoft.ML.Tokenizers;

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
            var firstPreparer = new TextBatchPreparer(tokenizer, sequence);
            var firstPreparation = Measure(() => firstPreparer.Prepare(["An initial uncached sentence."], includeTypes), 1);
            startup.Add(new { model, constructionUs, constructionBytes,
                firstPreparationUs = firstPreparation.Microseconds, firstPreparationBytes = firstPreparation.Bytes });
            foreach (int size in new[] { 1, 8, 32 })
            foreach (int length in new[] { 16, 128, limit })
            {
                string[] texts = Enumerable.Range(0, size)
                    .Select(i => string.Join(' ', Enumerable.Repeat("hello", length - 2 - i % 3))).ToArray();
                var preparer = new TextBatchPreparer(tokenizer, sequence);
                Func<object> baseline = () => PrepareBaseline(tokenizer, sequence, texts, includeTypes);
                Func<object> helper = () => preparer.Prepare(texts, includeTypes);
                TokenBatch prepared = preparer.Prepare(texts, includeTypes);
                TokenBatch simple = PrepareBaseline(tokenizer, sequence, texts, includeTypes);
                if (!simple.InputIds.Span.SequenceEqual(prepared.InputIds.Span) ||
                    !simple.AttentionMask.Span.SequenceEqual(prepared.AttentionMask.Span) ||
                    !simple.TokenTypeIds.Span.SequenceEqual(prepared.TokenTypeIds.Span))
                    throw new InvalidOperationException("Preparation benchmark implementations differ.");
                results.Add(new { stage = "prepare", model, batch = size, sequence = prepared.SequenceLength, tokenTypeIds = includeTypes,
                    measurements = Compare(baseline, helper, 30) });

                var hidden = new float[size * prepared.SequenceLength * 384];
                for (int i = 0; i < hidden.Length; i++) hidden[i] = MathF.Sin(i * 0.1f) + 0.1f;
                PoolingMode mode = model == "granite" ? PoolingMode.Cls : PoolingMode.Mean;
                Func<object> poolBaseline = () => PoolBaseline(hidden, prepared, 384, mode);
                Func<object> poolHelper = () => EmbeddingPooling.Pool(hidden, prepared, 384, mode);
                float[][] expected = PoolBaseline(hidden, prepared, 384, mode);
                float[][] actual = EmbeddingPooling.Pool(hidden, prepared, 384, mode);
                for (int row = 0; row < size; row++)
                    for (int column = 0; column < 384; column++)
                        if (MathF.Abs(expected[row][column] - actual[row][column]) > 2e-6f)
                            throw new InvalidOperationException("Pooling benchmark implementations differ.");
                results.Add(new { stage = $"{mode.ToString().ToLowerInvariant()}-pool-normalize", model, batch = size, sequence = prepared.SequenceLength,
                    measurements = Compare(poolBaseline, poolHelper, 30) });
            }
        }
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            runtime = RuntimeInformation.FrameworkDescription,
            os = RuntimeInformation.OSDescription,
            architecture = RuntimeInformation.ProcessArchitecture.ToString(),
            processors = Environment.ProcessorCount,
            tieredCompilation = false,
            implementation = "standard-tokenizer-v1",
            note = "Median of 7 alternating-order rounds, 30 calls per round, 10 warmups. Current-thread managed allocations. Cached repeated texts. No tokenizer construction, inference, native memory or input construction in steady-state measurements.",
            startup,
            results
        }, new JsonSerializerOptions { WriteIndented = true }));
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
        var mask = batch.AttentionMask.Span;
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
