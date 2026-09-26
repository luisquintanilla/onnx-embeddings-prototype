using System.Numerics.Tensors;
using CommunityToolkit.Embeddings.Onnx;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Tokenizers;
using Microsoft.Extensions.AI;

if (args.Length != 2 || args[0] is not ("minilm" or "e5" or "granite"))
    throw new ArgumentException("Usage: Composition <minilm|e5|granite> <local-model-directory>");

bool granite = args[0] == "granite";
Tokenizer tokenizer = granite
    ? new Granite30MEnglishTokenizer(Path.Combine(args[1], "vocab.json"), Path.Combine(args[1], "merges.txt"))
    : new BertUncasedTokenizer(Path.Combine(args[1], "vocab.txt"));
var sequence = new TokenSequenceOptions(args[0] == "minilm" ? 256 : 512,
    beginningTokenId: granite ? 0 : 101, endTokenId: granite ? 2 : 102, paddingTokenId: granite ? 1 : 0);
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

TokenBatch batch = tokenizer.PrepareBatch(texts, sequence, maximumBatchSize: 8, includeTokenTypeIds: encoder.RequiresTokenTypeIds);
Tensor<float> hiddenStates = encoder.Score(batch);
Tensor<float> vectors = EmbeddingPooling.Pool(hiddenStates, batch.AttentionMask, granite ? PoolingMode.Cls : PoolingMode.Mean);
int dimensions = checked((int)vectors.Lengths[1]);

Console.WriteLine($"Prepared [{batch.BatchSize}, {batch.SequenceLength}] -> [{string.Join(", ", hiddenStates.Lengths.ToArray())}] -> [{string.Join(", ", vectors.Lengths.ToArray())}]");
Console.WriteLine($"Dog / puppy:    {TensorPrimitives.CosineSimilarity<float>(vectors.GetSpan([0, 0], dimensions), vectors.GetSpan([1, 0], dimensions)):F4}");
Console.WriteLine($"Dog / database: {TensorPrimitives.CosineSimilarity<float>(vectors.GetSpan([0, 0], dimensions), vectors.GetSpan([2, 0], dimensions)):F4}");

// Copy from caller-mutable tensor storage when adapting final vectors to MEAI.
var embeddings = new GeneratedEmbeddings<Embedding<float>>();
for (int row = 0; row < vectors.Lengths[0]; row++)
    embeddings.Add(new Embedding<float>(vectors.GetSpan([row, 0], dimensions).ToArray()));
Console.WriteLine($"{embeddings.Count} owned final MEAI embeddings; numerical stages need neither MEAI nor ML.NET types.");
