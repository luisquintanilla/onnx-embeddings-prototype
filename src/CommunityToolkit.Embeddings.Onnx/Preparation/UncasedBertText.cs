using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace CommunityToolkit.Embeddings.Onnx;

// Exact HF basic-text/added-token policy for the two pinned uncased BERT vocabularies.
// Microsoft.ML.Tokenizers remains responsible for WordPiece segmentation and vocabulary lookup.
internal sealed partial class UncasedBertText
{
    private readonly BertTokenizer _tokenizer;
    private readonly Dictionary<string, int> _specialTokens;

    [GeneratedRegex(@"\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|\[MASK\]", RegexOptions.CultureInvariant, 30000)]
    private static partial Regex Specials();

    public UncasedBertText(string vocabularyPath)
    {
        using var stream = File.OpenRead(vocabularyPath);
        _tokenizer = BertTokenizer.Create(stream, new BertOptions { ApplyBasicTokenization = false, SplitOnSpecialTokens = false });
        _specialTokens = new(StringComparer.Ordinal)
        {
            ["[PAD]"] = _tokenizer.PaddingTokenId,
            ["[UNK]"] = _tokenizer.UnknownTokenId,
            ["[CLS]"] = _tokenizer.ClassificationTokenId,
            ["[SEP]"] = _tokenizer.SeparatorTokenId,
            ["[MASK]"] = _tokenizer.MaskingTokenId
        };
    }

    public int BeginningTokenId => _tokenizer.ClassificationTokenId;
    public int EndTokenId => _tokenizer.SeparatorTokenId;
    public int PaddingTokenId => _tokenizer.PaddingTokenId;

    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>();
        int position = 0;
        // HF's normalized=false added tokens are recognized before ordinary text is lowercased.
        foreach (Match special in Specials().Matches(text))
        {
            EncodeOrdinary(text[position..special.Index], ids);
            ids.Add(_specialTokens[special.Value]);
            position = special.Index + special.Length;
        }
        EncodeOrdinary(text[position..], ids);
        return ids;
    }

    private void EncodeOrdinary(string text, List<int> ids)
    {
        string normalized = Normalize(text);
        int start = 0;
        int position = 0;
        foreach (Rune rune in normalized.EnumerateRunes())
        {
            int width = rune.Utf16SequenceLength;
            bool whitespace = Rune.IsWhiteSpace(rune);
            bool punctuation = IsPunctuation(rune);
            if (whitespace || punctuation)
            {
                EncodeWord(normalized.AsSpan(start, position - start), ids);
                if (punctuation) EncodeWord(normalized.AsSpan(position, width), ids);
                start = position + width;
            }
            position += width;
        }
        EncodeWord(normalized.AsSpan(start), ids);
    }

    private void EncodeWord(ReadOnlySpan<char> word, List<int> ids)
    {
        if (!word.IsEmpty)
            ids.AddRange(_tokenizer.EncodeToIds(word, addSpecialTokens: false,
                considerPreTokenization: false, considerNormalization: false));
    }

    private static string Normalize(string text)
    {
        var clean = new StringBuilder(text.Length);
        foreach (Rune rune in text.EnumerateRunes())
        {
            if (rune.Value is '\t' or '\n' or '\r') { clean.Append(' '); continue; }
            if (rune.Value is 0 or 0xFFFD || Rune.GetUnicodeCategory(rune) is
                UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.PrivateUse or UnicodeCategory.OtherNotAssigned)
                continue;
            if (Rune.IsWhiteSpace(rune)) { clean.Append(' '); continue; }
            bool chinese = IsChinese(rune.Value);
            if (chinese) clean.Append(' ');
            AppendRune(clean, rune);
            if (chinese) clean.Append(' ');
        }

        string decomposed = clean.ToString().Normalize(NormalizationForm.FormD);
        clean.Clear();
        foreach (Rune rune in decomposed.EnumerateRunes())
            if (Rune.GetUnicodeCategory(rune) != UnicodeCategory.NonSpacingMark)
                AppendRune(clean, Rune.ToLowerInvariant(rune));
        // HF leaves the stripped text decomposed; recomposing can change WordPiece IDs (e.g. Hangul).
        return clean.ToString();
    }

    private static void AppendRune(StringBuilder builder, Rune rune)
    {
        Span<char> buffer = stackalloc char[2];
        int written = rune.EncodeToUtf16(buffer);
        builder.Append(buffer[..written]);
    }

    private static bool IsChinese(int value) => value is
        >= 0x4E00 and <= 0x9FFF or >= 0x3400 and <= 0x4DBF or >= 0x20000 and <= 0x2A6DF or
        >= 0x2A700 and <= 0x2B73F or >= 0x2B740 and <= 0x2B81F or >= 0x2B920 and <= 0x2CEAF or
        >= 0xF900 and <= 0xFAFF or >= 0x2F800 and <= 0x2FA1F;

    private static bool IsPunctuation(Rune rune) => rune.Value is
        >= 33 and <= 47 or >= 58 and <= 64 or >= 91 and <= 96 or >= 123 and <= 126 ||
        Rune.GetUnicodeCategory(rune) is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation or
        UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation or UnicodeCategory.InitialQuotePunctuation or
        UnicodeCategory.FinalQuotePunctuation or UnicodeCategory.OtherPunctuation;
}
