using System.Text;
using System.Text.Json;
using CommunityToolkit.Embeddings.Onnx;
using Microsoft.ML.Tokenizers;

if (args.Length == 2 && args[0] == "measure")
{
    Measurements.Run(args[1]);
    return;
}
if (args.Length == 2 && args[0] == "bert")
{
    var builtinBert = BertTokenizer.Create(Path.Combine(args[1], "vocab.txt"),
        new BertOptions { RemoveNonSpacingMarks = true, SplitOnSpecialTokens = true });
    var adaptedBert = TextTokenizer.CreateUncasedBert(Path.Combine(args[1], "vocab.txt"));
    using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(args[1], "reference.json")));
    foreach (var batch in reference.RootElement.GetProperty("batches").EnumerateArray())
    {
        var items = batch.GetProperty("items");
        for (int row = 0; row < items.GetArrayLength(); row++)
        {
            string text = items[row].GetProperty("text").GetString()!;
            if (text.Length > 80) continue;
            string? purpose = items[row].GetProperty("purpose").GetString();
            if (purpose is not null) text = E5Text.Format(text, Enum.Parse<E5Purpose>(purpose));
            int length = batch.GetProperty("attentionMask")[row].EnumerateArray().Sum(x => x.GetInt32());
            int[] expected = batch.GetProperty("inputIds")[row].EnumerateArray().Take(length - 1).Skip(1).Select(x => x.GetInt32()).ToArray();
            int[] direct = builtinBert.EncodeToIds(text, addSpecialTokens: false).ToArray();
            int[] actual = adaptedBert.EncodeContent(text).ToArray();
            Console.WriteLine(JsonSerializer.Serialize(new { text, expected, direct, adapter = actual }));
            if (!expected.SequenceEqual(actual)) throw new InvalidOperationException("BERT adapter token mismatch.");
        }
    }
    return;
}
if (args.Length != 2 || args[0] != "roberta")
    throw new ArgumentException("Usage: Experiments <bert|roberta> <model-directory> | measure <asset-root>");

string directory = args[1];
using var vocab = File.OpenRead(Path.Combine(directory, "vocab.json"));
using var merges = File.OpenRead(Path.Combine(directory, "merges.txt"));
// Neutralize Fairseq ID remapping: HF vocab IDs >=4 map to their own ranks.
using var mapping = new MemoryStream(Encoding.UTF8.GetBytes(
    string.Join('\n', Enumerable.Range(4, 50261).Select(id => $"{id} 1"))));
var builtin = EnglishRobertaTokenizer.Create(vocab, merges, mapping);
var adapter = TextTokenizer.CreateGranite30MEnglish(Path.Combine(directory, "vocab.json"), Path.Combine(directory, "merges.txt"));
using var probes = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "tokenizer-probes.json")));
foreach (var probe in probes.RootElement.EnumerateArray())
{
    string text = probe.GetProperty("text").GetString()!;
    int[] expected = probe.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray();
    int[] direct = builtin.EncodeToIds(text).ToArray();
    int[] utf8 = builtin.EncodeToIds(Encoding.Latin1.GetString(Encoding.UTF8.GetBytes(text))).ToArray();
    int[] actual = adapter.EncodeContent(text).ToArray();
    Console.WriteLine(JsonSerializer.Serialize(new { text, expected, direct, utf8, adapter = actual }));
    if (!expected.SequenceEqual(actual)) throw new InvalidOperationException("Adapter token mismatch.");
}
