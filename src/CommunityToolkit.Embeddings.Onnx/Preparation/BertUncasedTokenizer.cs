using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace CommunityToolkit.Embeddings.Onnx;

/// <summary>Content-only WordPiece tokenizer with the pinned MiniLM/E5 uncased Hugging Face text policy.</summary>
/// <remarks>
/// Offsets and consumption refer to NormalizedText when normalization is enabled, otherwise the input.
/// Decode uses Microsoft's WordPiece span-decoder spacing, not original spelling or Hugging Face cleanup.
/// </remarks>
public sealed partial class BertUncasedTokenizer : Tokenizer
{
    private readonly BertTokenizer _tokenizer;

    [GeneratedRegex(@"\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|\[MASK\]", RegexOptions.CultureInvariant, 30000)]
    private static partial Regex Specials();

    public BertUncasedTokenizer(string vocabularyPath)
    {
        using var stream = File.OpenRead(vocabularyPath);
        _tokenizer = BertTokenizer.Create(stream, new BertOptions
        {
            ApplyBasicTokenization = false,
            SplitOnSpecialTokens = false,
            Normalizer = new UncasedNormalizer(),
            PreTokenizer = new BasicPreTokenizer()
        });
    }

    public int ClassificationTokenId => _tokenizer.ClassificationTokenId;
    public int SeparatorTokenId => _tokenizer.SeparatorTokenId;
    public int PaddingTokenId => _tokenizer.PaddingTokenId;
    public override Normalizer? Normalizer => _tokenizer.Normalizer;
    public override PreTokenizer? PreTokenizer => _tokenizer.PreTokenizer;

    protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(settings.MaxTokenCount, 1);
        ReadOnlySpan<char> input = text is null ? textSpan : text.AsSpan();
        TokenizerContract.ValidateText(input);
        var tokens = _tokenizer.EncodeToTokens(input, out string? normalized,
            settings.ConsiderPreTokenization, settings.ConsiderNormalization);
        return TokenizerContract.Limit(tokens, normalized, normalized?.Length ?? input.Length, settings.MaxTokenCount);
    }

    protected override int GetIndexByTokenCount(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings,
        bool fromEnd, out string? normalizedText, out int tokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(settings.MaxTokenCount, 1);
        int maximum = settings.MaxTokenCount;
        settings.MaxTokenCount = int.MaxValue;
        var encoded = EncodeToTokens(text, textSpan, settings);
        normalizedText = encoded.NormalizedText;
        return TokenizerContract.GetBoundary(encoded.Tokens, encoded.CharsConsumed, maximum, fromEnd, out tokenCount);
    }

    public override string Decode(IEnumerable<int> ids) => TokenizerContract.DecodeToString(this, ids);

    public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten)
    {
        ArgumentNullException.ThrowIfNull(ids);
        return _tokenizer.Decode(ids, destination, out idsConsumed, out charsWritten);
    }

    private sealed class UncasedNormalizer : Normalizer
    {
        public override string Normalize(string original) => Normalize(original.AsSpan());

        public override string Normalize(ReadOnlySpan<char> original)
        {
            TokenizerContract.ValidateText(original);
            string text = original.ToString();
            var result = new StringBuilder(text.Length);
            int position = 0;
            // Preserve normalized=false added tokens before ordinary text is lowercased.
            foreach (Match special in Specials().Matches(text))
            {
                result.Append(NormalizeOrdinary(text[position..special.Index]));
                result.Append(special.Value);
                position = special.Index + special.Length;
            }
            result.Append(NormalizeOrdinary(text[position..]));
            return result.ToString();
        }
    }

    private sealed class BasicPreTokenizer : PreTokenizer
    {
        public override IEnumerable<(int Offset, int Length)> PreTokenize(ReadOnlySpan<char> text) => PreTokenize(text.ToString());

        public override IEnumerable<(int Offset, int Length)> PreTokenize(string text)
        {
            TokenizerContract.ValidateText(text);
            var pieces = new List<(int, int)>();
            int position = 0;
            foreach (Match special in Specials().Matches(text))
            {
                SplitOrdinary(text.AsSpan(position, special.Index - position), position, pieces);
                pieces.Add((special.Index, special.Length));
                position = special.Index + special.Length;
            }
            SplitOrdinary(text.AsSpan(position), position, pieces);
            return pieces;
        }

        private static void SplitOrdinary(ReadOnlySpan<char> text, int offset, List<(int, int)> pieces)
        {
            int start = 0;
            int position = 0;
            foreach (Rune rune in text.EnumerateRunes())
            {
                int width = rune.Utf16SequenceLength;
                bool punctuation = IsPunctuation(rune);
                if (Rune.IsWhiteSpace(rune) || punctuation)
                {
                    if (position > start) pieces.Add((offset + start, position - start));
                    if (punctuation) pieces.Add((offset + position, width));
                    start = position + width;
                }
                position += width;
            }
            if (position > start) pieces.Add((offset + start, position - start));
        }
    }

    private static string NormalizeOrdinary(string text)
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
