namespace CommunityToolkit.Embeddings.Onnx;

/// <summary>Immutable single-sequence framing: one beginning token, content, one end token, then right padding.</summary>
public sealed class TokenSequenceOptions
{
    public TokenSequenceOptions(int maximumSequenceLength, int beginningTokenId, int endTokenId, int paddingTokenId)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maximumSequenceLength, 2);
        ArgumentOutOfRangeException.ThrowIfNegative(beginningTokenId);
        ArgumentOutOfRangeException.ThrowIfNegative(endTokenId);
        ArgumentOutOfRangeException.ThrowIfNegative(paddingTokenId);
        MaximumSequenceLength = maximumSequenceLength;
        BeginningTokenId = beginningTokenId;
        EndTokenId = endTokenId;
        PaddingTokenId = paddingTokenId;
    }

    public int MaximumSequenceLength { get; }
    public int BeginningTokenId { get; }
    public int EndTokenId { get; }
    public int PaddingTokenId { get; }
}
