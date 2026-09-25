using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace CommunityToolkit.Embeddings.Onnx;

// HF byte-level preprocessing only. Microsoft.ML.Tokenizers owns the BPE engine.
internal sealed partial class GraniteByteBpe
{
    private readonly BpeTokenizer _bpe;
    private static readonly char[] ByteAlphabet = CreateByteAlphabet();
    private static readonly Dictionary<string, int> AddedTokens = new(StringComparer.Ordinal)
    {
        ["<s>"] = 0, ["<pad>"] = 1, ["</s>"] = 2, ["<unk>"] = 3, ["<mask>"] = 50264
    };

    [GeneratedRegex(@"<s>|<pad>|</s>|<unk>|<mask>", RegexOptions.CultureInvariant, 30000)]
    private static partial Regex Specials();

    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.CultureInvariant, 30000)]
    private static partial Regex Pieces();

    public GraniteByteBpe(string vocabularyPath, string mergesPath)
    {
        using var vocabulary = File.OpenRead(vocabularyPath);
        using var merges = File.OpenRead(mergesPath);
        _bpe = BpeTokenizer.Create(vocabulary, merges, preTokenizer: null, normalizer: null, unknownToken: "<unk>");
        foreach (var token in AddedTokens)
        {
            if (!_bpe.Vocabulary.TryGetValue(token.Key, out int id) || id != token.Value)
                throw new ArgumentException($"Granite requires {token.Key} to have token ID {token.Value}.");
        }
    }

    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>();
        int position = 0;
        foreach (Match special in Specials().Matches(text))
        {
            int end = special.Index;
            // Granite's AddedToken("<mask>", lstrip=true) consumes preceding Unicode whitespace.
            if (special.Value == "<mask>")
                while (end > position && char.IsWhiteSpace(text[end - 1])) end--;
            EncodeOrdinary(text[position..end], ids);
            ids.Add(AddedTokens[special.Value]);
            position = special.Index + special.Length;
        }
        EncodeOrdinary(text[position..], ids);
        return ids;
    }

    private void EncodeOrdinary(string text, List<int> ids)
    {
        // .NET regex categories are UTF-16 based. Preserve offsets but classify astral letters/numbers
        // as letters/numbers, so the HF Unicode-regex boundaries also hold for supplementary planes.
        string classified = ClassifySupplementary(text);
        foreach (Match piece in Pieces().Matches(classified))
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text.AsSpan(piece.Index, piece.Length).ToString());
            string encoded = string.Create(bytes.Length, bytes, static (target, source) =>
            {
                for (int i = 0; i < source.Length; i++) target[i] = ByteAlphabet[source[i]];
            });
            ids.AddRange(_bpe.EncodeToIds(encoded, considerPreTokenization: false, considerNormalization: false));
        }
    }

    private static string ClassifySupplementary(string text)
    {
        char[]? classified = null;
        for (int i = 0; i + 1 < text.Length; i++)
        {
            if (!Rune.TryCreate(text[i], text[i + 1], out Rune rune)) continue;
            UnicodeCategory category = Rune.GetUnicodeCategory(rune);
            char replacement = category is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter
                or UnicodeCategory.TitlecaseLetter or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter ? 'A'
                : category is UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber ? '1' : '\0';
            if (replacement != '\0')
            {
                classified ??= text.ToCharArray();
                classified[i] = classified[i + 1] = replacement;
            }
            i++;
        }
        return classified is null ? text : new string(classified);
    }

    private static char[] CreateByteAlphabet()
    {
        var alphabet = new char[256];
        int extra = 256;
        for (int i = 0; i < alphabet.Length; i++)
            alphabet[i] = i is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255 ? (char)i : (char)extra++;
        return alphabet;
    }
}
