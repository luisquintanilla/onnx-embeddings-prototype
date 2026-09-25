using System.Buffers;
using System.Text;

namespace CommunityToolkit.Embeddings.Onnx;

/// <summary>A content-token encoder and the single-sequence special-token IDs it uses.</summary>
public sealed class TextTokenizer
{
    private readonly Func<string, IReadOnlyList<int>> _encode;

    public TextTokenizer(Func<string, IReadOnlyList<int>> encodeContent, int beginningTokenId, int endTokenId, int paddingTokenId)
    {
        ArgumentNullException.ThrowIfNull(encodeContent);
        ArgumentOutOfRangeException.ThrowIfNegative(beginningTokenId);
        ArgumentOutOfRangeException.ThrowIfNegative(endTokenId);
        ArgumentOutOfRangeException.ThrowIfNegative(paddingTokenId);
        _encode = encodeContent;
        BeginningTokenId = beginningTokenId;
        EndTokenId = endTokenId;
        PaddingTokenId = paddingTokenId;
    }

    public int BeginningTokenId { get; }
    public int EndTokenId { get; }
    public int PaddingTokenId { get; }

    /// <summary>Encodes content only, including literal added tokens but not surrounding BOS/EOS.</summary>
    public IReadOnlyList<int> EncodeContent(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        ReadOnlySpan<char> remaining = text;
        while (!remaining.IsEmpty)
        {
            if (Rune.DecodeFromUtf16(remaining, out _, out int consumed) != OperationStatus.Done)
                throw new ArgumentException("Text must contain well-formed UTF-16.", nameof(text));
            remaining = remaining[consumed..];
        }
        return _encode(text) ?? throw new InvalidOperationException("The content encoder returned null.");
    }

    public static TextTokenizer CreateUncasedBert(string vocabularyPath)
    {
        var adapter = new UncasedBertText(vocabularyPath);
        return new TextTokenizer(adapter.Encode, adapter.BeginningTokenId, adapter.EndTokenId, adapter.PaddingTokenId);
    }

    public static TextTokenizer CreateGranite30MEnglish(string vocabularyPath, string mergesPath)
    {
        var adapter = new GraniteByteBpe(vocabularyPath, mergesPath);
        return new TextTokenizer(adapter.Encode, 0, 2, 1);
    }
}
