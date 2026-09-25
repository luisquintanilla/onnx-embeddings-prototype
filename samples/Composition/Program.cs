using System.Numerics.Tensors;
using CommunityToolkit.Embeddings.Onnx;
using Microsoft.ML.OnnxRuntime;

if (args.Length != 2 || args[0] is not ("minilm" or "e5" or "granite"))
    throw new ArgumentException("Usage: Composition <minilm|e5|granite> <local-model-directory>");

bool granite = args[0] == "granite";
TextTokenizer tokenizer = granite
    ? TextTokenizer.CreateGranite30MEnglish(Path.Combine(args[1], "vocab.json"), Path.Combine(args[1], "merges.txt"))
    : TextTokenizer.CreateUncasedBert(Path.Combine(args[1], "vocab.txt"));
var preparer = new TextBatchPreparer(tokenizer, args[0] == "minilm" ? 256 : 512, maximumBatchSize: 8);
using var options = new SessionOptions { IntraOpNumThreads = 2 };
using var encoder = OnnxTextEncoder.Load(Path.Combine(args[1], "model.onnx"), options,
    outputName: granite ? "logits" : "last_hidden_state");
string[] texts =
[
    "A dog is playing in the park.",
    "A puppy plays outdoors.",
    "Database indexes accelerate queries."
];
if (args[0] == "e5")
    texts = [E5Text.Format(texts[0], E5Purpose.Query), .. texts.Skip(1).Select(text => E5Text.Format(text, E5Purpose.Document))];

TokenBatch batch = preparer.Prepare(texts, encoder.RequiresTokenTypeIds);
float[] hiddenStates = encoder.Score(batch);
float[][] vectors = EmbeddingPooling.Pool(hiddenStates, batch, encoder.Dimensions, granite ? PoolingMode.Cls : PoolingMode.Mean);

Console.WriteLine($"Prepared [{batch.BatchSize}, {batch.SequenceLength}] -> token states -> {vectors.Length} x {vectors[0].Length}");
Console.WriteLine($"Dog / puppy:    {TensorPrimitives.CosineSimilarity<float>(vectors[0], vectors[1]):F4}");
Console.WriteLine($"Dog / database: {TensorPrimitives.CosineSimilarity<float>(vectors[0], vectors[2]):F4}");
Console.WriteLine("This independent assembly uses only the public stages, not MEAI or ML.NET APIs.");
