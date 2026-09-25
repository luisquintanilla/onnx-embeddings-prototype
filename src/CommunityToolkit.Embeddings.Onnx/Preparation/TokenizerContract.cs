using System.Buffers;
using System.Text;
using Microsoft.ML.Tokenizers;

namespace CommunityToolkit.Embeddings.Onnx;

internal static class TokenizerContract
{
    internal static void ValidateText(ReadOnlySpan<char> text)
    {
        while (!text.IsEmpty)
        {
            if (Rune.DecodeFromUtf16(text, out _, out int consumed) != OperationStatus.Done)
                throw new ArgumentException("Text must contain well-formed UTF-16.", nameof(text));
            text = text[consumed..];
        }
    }

    internal static EncodeResults<EncodedToken> Limit(
        IReadOnlyList<EncodedToken> tokens, string? normalizedText, int textLength, int maxTokenCount)
    {
        int end = GetBoundary(tokens, textLength, maxTokenCount, false, out int count);
        return new()
        {
            Tokens = count == tokens.Count ? tokens : tokens.Take(count).ToArray(),
            NormalizedText = normalizedText,
            CharsConsumed = end
        };
    }

    internal static int GetBoundary(
        IReadOnlyList<EncodedToken> tokens, int textLength, int maxTokenCount, bool fromEnd, out int count)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxTokenCount, 1);
        count = Math.Min(maxTokenCount, tokens.Count);
        if (count == tokens.Count) return fromEnd ? 0 : textLength;

        // Byte tokens can share a source scalar. A UTF-16 index cannot represent a partial scalar.
        if (fromEnd)
        {
            int start = tokens.Count - count;
            while (start < tokens.Count && tokens[start - 1].Offset.End.Value > tokens[start].Offset.Start.Value)
                start++;
            count = tokens.Count - start;
            return count == 0 ? textLength : tokens[start].Offset.Start.Value;
        }

        while (count > 0 && tokens[count - 1].Offset.End.Value > tokens[count].Offset.Start.Value)
            count--;
        return count == 0 ? 0 : tokens[count - 1].Offset.End.Value;
    }

    internal static string DecodeToString(Tokenizer tokenizer, IEnumerable<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        int[] snapshot = ids.ToArray();
        var buffer = new char[Math.Max(16, checked(snapshot.Length * 8))];
        while (true)
        {
            OperationStatus status = tokenizer.Decode(snapshot, buffer, out _, out int written);
            if (status == OperationStatus.Done) return new string(buffer, 0, written);
            if (status != OperationStatus.DestinationTooSmall)
                throw new InvalidOperationException("The provided token IDs could not be decoded.");
            buffer = new char[checked(buffer.Length * 2)];
        }
    }
}
