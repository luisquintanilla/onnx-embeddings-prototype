using System.Text.Json;
using CommunityToolkit.Embeddings.Onnx;
using Microsoft.ML.Tokenizers;

internal static class TokenizerContractProbe
{
    public static void Run(string assetRoot)
    {
        string granite = Path.Combine(assetRoot, "granite");
        Tokenizer adapter = new Granite30MEnglishTokenizer(Path.Combine(granite, "vocab.json"), Path.Combine(granite, "merges.txt"));
        Tokenizer builtin = BpeTokenizer.Create(new BpeOptions(Path.Combine(granite, "vocab.json"), Path.Combine(granite, "merges.txt"))
        {
            ByteLevel = true,
            PreTokenizer = adapter.PreTokenizer,
            SpecialTokens = new Dictionary<string, int>
            {
                ["<s>"] = 0, ["<pad>"] = 1, ["</s>"] = 2, ["<unk>"] = 3, ["<mask>"] = 50264
            }
        });
        foreach (string text in new[] { "\u00e9", "\U0001f600", "A\U0001f600 B", "  <mask>", "\U0001f600<s>" })
        {
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                kind = "granite-offsets",
                text,
                builtin = Tokens(builtin, text),
                adapter = Tokens(adapter, text)
            }));
        }
        int[] emoji = adapter.EncodeToIds("\U0001f600").ToArray();
        foreach (int[] ids in new[] { emoji, emoji[..1], emoji[1..], new[] { -1 } })
        foreach (int capacity in new[] { 0, 1, 2, 32 })
        {
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                kind = "granite-decode",
                ids,
                capacity,
                builtin = Decode(builtin, ids, capacity),
                adapter = Decode(adapter, ids, capacity)
            }));
        }
        Tokenizer bert = BertTokenizer.Create(Path.Combine(assetRoot, "minilm", "vocab.txt"));
        Tokenizer bertAdapter = new BertUncasedTokenizer(Path.Combine(assetRoot, "minilm", "vocab.txt"));
        int[] punctuation = bertAdapter.EncodeToIds("hello.").ToArray();
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            kind = "bert-decode",
            ids = punctuation,
            builtinString = bert.Decode(punctuation),
            builtinSpan = Decode(bert, punctuation, 32),
            adapterString = bertAdapter.Decode(punctuation),
            adapterSpan = Decode(bertAdapter, punctuation, 32)
        }));
    }

    private static object Tokens(Tokenizer tokenizer, string text)
        => tokenizer.EncodeToTokens(text, out _).Select(token =>
            new { token.Id, token.Value, start = token.Offset.Start.Value, end = token.Offset.End.Value }).ToArray();

    private static object Decode(Tokenizer tokenizer, int[] ids, int capacity)
    {
        var destination = new char[capacity];
        var status = tokenizer.Decode(ids, destination, out int consumed, out int written);
        return new { status = status.ToString(), consumed, written, text = new string(destination, 0, written) };
    }
}
