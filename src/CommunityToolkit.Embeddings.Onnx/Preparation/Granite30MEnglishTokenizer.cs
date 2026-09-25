using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace CommunityToolkit.Embeddings.Onnx;

/// <summary>Content-only byte BPE for the pinned Granite embedding 30M English tokenizer.</summary>
/// <remarks>
/// Offsets cover original UTF-16 scalars and may overlap when several byte tokens encode one scalar.
/// Bounded encoding keeps overlapping tokens together; batch preparation separately applies the model's token budget.
/// Decode is strict UTF-8: invalid IDs or incomplete byte sequences fail rather than silently losing text.
/// </remarks>
public sealed partial class Granite30MEnglishTokenizer : Tokenizer
{
    private readonly BpeTokenizer _bpe;
    private readonly IReadOnlyDictionary<int, string> _vocabulary;
    private static readonly IReadOnlyDictionary<char, byte> InverseByteAlphabet = CreateInverseByteAlphabet();
    private static readonly Dictionary<string, int> AddedTokens = new(StringComparer.Ordinal)
    {
        ["<s>"] = 0, ["<pad>"] = 1, ["</s>"] = 2, ["<unk>"] = 3, ["<mask>"] = 50264
    };

    [GeneratedRegex(@"<s>|<pad>|</s>|<unk>|<mask>", RegexOptions.CultureInvariant, 30000)]
    private static partial Regex Specials();

    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.CultureInvariant, 30000)]
    private static partial Regex Pieces();

    public Granite30MEnglishTokenizer(string vocabularyPath, string mergesPath)
    {
        _bpe = BpeTokenizer.Create(new BpeOptions(vocabularyPath, mergesPath)
        {
            ByteLevel = true,
            PreTokenizer = new BytePreTokenizer(),
            SpecialTokens = AddedTokens,
            UnknownToken = "<unk>"
        });
        foreach (var token in AddedTokens)
        {
            if (!_bpe.Vocabulary.TryGetValue(token.Key, out int id) || id != token.Value)
                throw new ArgumentException($"Granite requires {token.Key} to have token ID {token.Value}.");
        }
        _vocabulary = _bpe.Vocabulary.ToDictionary(pair => pair.Value, pair => pair.Key);
    }

    public override PreTokenizer? PreTokenizer => _bpe.PreTokenizer;

    protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(settings.MaxTokenCount, 1);
        ReadOnlySpan<char> input = text is null ? textSpan : text.AsSpan();
        TokenizerContract.ValidateText(input);
        var encoded = _bpe.EncodeToTokens(input, out _, settings.ConsiderPreTokenization, settings.ConsiderNormalization);
        var tokens = new List<EncodedToken>(encoded.Count);
        foreach (EncodedToken token in encoded)
        {
            int start = token.Offset.Start.Value;
            int end = token.Offset.End.Value;
            // The pinned byte engine maps the last byte to the first UTF-16 code unit plus one.
            if (start > 0 && start < input.Length && char.IsLowSurrogate(input[start])) start--;
            if (end > 0 && end < input.Length && char.IsLowSurrogate(input[end])) end++;
            tokens.Add(new EncodedToken(token.Id, token.Value, start..end));
        }
        return TokenizerContract.Limit(tokens, null, input.Length, settings.MaxTokenCount);
    }

    protected override int GetIndexByTokenCount(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings,
        bool fromEnd, out string? normalizedText, out int tokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(settings.MaxTokenCount, 1);
        int maximum = settings.MaxTokenCount;
        settings.MaxTokenCount = int.MaxValue;
        var encoded = EncodeToTokens(text, textSpan, settings);
        normalizedText = null;
        return TokenizerContract.GetBoundary(encoded.Tokens, encoded.CharsConsumed, maximum, fromEnd, out tokenCount);
    }

    public override string Decode(IEnumerable<int> ids) => TokenizerContract.DecodeToString(this, ids);

    public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten)
    {
        ArgumentNullException.ThrowIfNull(ids);
        idsConsumed = charsWritten = 0;
        int pendingIds = 0;
        var bytes = new ArrayBufferWriter<byte>();
        foreach (int id in ids)
        {
            if (!_vocabulary.TryGetValue(id, out string? value)) return OperationStatus.InvalidData;
            Span<byte> target = bytes.GetSpan(value.Length);
            for (int i = 0; i < value.Length; i++)
            {
                if (!InverseByteAlphabet.TryGetValue(value[i], out target[i])) return OperationStatus.InvalidData;
            }
            bytes.Advance(value.Length);
            pendingIds++;

            ReadOnlySpan<byte> remaining = bytes.WrittenSpan;
            while (!remaining.IsEmpty)
            {
                OperationStatus status = Rune.DecodeFromUtf8(remaining, out _, out int consumed);
                if (status == OperationStatus.NeedMoreData) break;
                if (status != OperationStatus.Done) return OperationStatus.InvalidData;
                remaining = remaining[consumed..];
            }
            if (!remaining.IsEmpty) continue;

            int count = Encoding.UTF8.GetCharCount(bytes.WrittenSpan);
            if (count > destination.Length - charsWritten) return OperationStatus.DestinationTooSmall;
            charsWritten += Encoding.UTF8.GetChars(bytes.WrittenSpan, destination[charsWritten..]);
            idsConsumed += pendingIds;
            pendingIds = 0;
            bytes.Clear();
        }
        return pendingIds == 0 ? OperationStatus.Done : OperationStatus.InvalidData;
    }

    private sealed class BytePreTokenizer : PreTokenizer
    {
        public override IEnumerable<(int Offset, int Length)> PreTokenize(ReadOnlySpan<char> text) => PreTokenize(text.ToString());

        public override IEnumerable<(int Offset, int Length)> PreTokenize(string text)
        {
            TokenizerContract.ValidateText(text);
            var pieces = new List<(int, int)>();
            int position = 0;
            foreach (Match special in Specials().Matches(text))
            {
                int end = special.Index;
                // AddedToken("<mask>", lstrip=true); its offset covers the literal, not discarded whitespace.
                if (special.Value == "<mask>")
                    while (end > position && char.IsWhiteSpace(text[end - 1])) end--;
                SplitOrdinary(text[position..end], position, pieces);
                pieces.Add((special.Index, special.Length));
                position = special.Index + special.Length;
            }
            SplitOrdinary(text[position..], position, pieces);
            return pieces;
        }

        private static void SplitOrdinary(string text, int offset, List<(int, int)> pieces)
        {
            // .NET regex categories are UTF-16 based; classify astral letters/numbers without moving offsets.
            string classified = ClassifySupplementary(text);
            foreach (Match piece in Pieces().Matches(classified))
                pieces.Add((offset + piece.Index, piece.Length));
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

    private static IReadOnlyDictionary<char, byte> CreateInverseByteAlphabet()
    {
        var alphabet = new Dictionary<char, byte>(256);
        int extra = 256;
        for (int i = 0; i < 256; i++)
        {
            char value = i is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255 ? (char)i : (char)extra++;
            alphabet.Add(value, (byte)i);
        }
        return alphabet;
    }
}
