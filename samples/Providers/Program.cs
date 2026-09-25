using System.Numerics.Tensors;
using CommunityToolkit.Embeddings.Onnx;
using Microsoft.Extensions.AI;
using Microsoft.ML.OnnxRuntime;

if (args.Length != 2 || args[0] is not ("minilm" or "e5" or "granite"))
    throw new ArgumentException("Usage: Providers <minilm|e5|granite> <local-model-directory>");

using var options = new SessionOptions { IntraOpNumThreads = 2 };
using IEmbeddingGenerator<string, Embedding<float>> generator = args[0] switch
{
    "minilm" => new AllMiniLmL6V2EmbeddingGenerator(args[1], options),
    "e5" => new E5SmallV2EmbeddingGenerator(args[1], E5Purpose.Query, options),
    _ => new GraniteEmbedding30MEnglishGenerator(args[1], options)
};

string query = "How can I make database searches faster?";
string[] documents =
[
    "Database indexes help speed up queries by avoiding full table scans.",
    "A puppy is playing with a ball in the park.",
    "Bread dough rises when yeast ferments sugars."
];
GeneratedEmbeddings<Embedding<float>> embeddings;
if (generator is E5SmallV2EmbeddingGenerator e5)
{
    E5Input[] inputs = [new(query, E5Purpose.Query), .. documents.Select(text => new E5Input(text, E5Purpose.Document))];
    embeddings = await e5.GenerateAsync(inputs);
}
else
{
    embeddings = await generator.GenerateAsync([query, .. documents]);
}

Console.WriteLine($"Query: {query}");
foreach (var match in documents.Select((text, index) => new
{
    Text = text,
    Score = TensorPrimitives.CosineSimilarity<float>(embeddings[0].Vector.Span, embeddings[index + 1].Vector.Span)
}).OrderByDescending(match => match.Score))
    Console.WriteLine($"{match.Score:F4}  {match.Text}");
Console.WriteLine($"{embeddings.Count} vectors, {embeddings[0].Vector.Length} dimensions; local assets only.");
