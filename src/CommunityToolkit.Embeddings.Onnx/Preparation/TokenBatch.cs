using System.Numerics.Tensors;

namespace CommunityToolkit.Embeddings.Onnx;

/// <summary>Row-major, right-padded Int64 tensors. The batch owns its managed buffers.</summary>
public sealed class TokenBatch
{
    internal readonly long[] IdBuffer;
    internal readonly long[] MaskBuffer;
    internal readonly long[]? TypeBuffer;

    public TokenBatch(int batchSize, int sequenceLength, ReadOnlySpan<long> inputIds,
        ReadOnlySpan<long> attentionMask, ReadOnlySpan<long> tokenTypeIds = default)
        : this(batchSize, sequenceLength, inputIds.ToArray(), attentionMask.ToArray(),
            tokenTypeIds.IsEmpty ? null : tokenTypeIds.ToArray())
    {
    }

    internal TokenBatch(int batchSize, int sequenceLength, long[] inputIds, long[] attentionMask, long[]? tokenTypeIds)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(batchSize);
        ArgumentOutOfRangeException.ThrowIfNegative(sequenceLength);
        if ((batchSize == 0) != (sequenceLength == 0))
            throw new ArgumentException("An empty batch must have both dimensions zero.");
        int count = checked(batchSize * sequenceLength);
        if (inputIds.Length != count || attentionMask.Length != count || (tokenTypeIds is not null && tokenTypeIds.Length != count))
            throw new ArgumentException("Tensor buffers must match batchSize * sequenceLength.");
        for (int i = 0; i < count; i++)
        {
            if (inputIds[i] < 0 || attentionMask[i] is not (0 or 1) || (tokenTypeIds is not null && tokenTypeIds[i] < 0))
                throw new ArgumentException("IDs must be nonnegative and attention masks must contain only zero or one.");
        }
        BatchSize = batchSize;
        SequenceLength = sequenceLength;
        IdBuffer = inputIds;
        MaskBuffer = attentionMask;
        TypeBuffer = tokenTypeIds;
    }

    public int BatchSize { get; }
    public int SequenceLength { get; }
    public ReadOnlyTensorSpan<long> InputIds => new(IdBuffer, [BatchSize, SequenceLength]);
    public ReadOnlyTensorSpan<long> AttentionMask => new(MaskBuffer, [BatchSize, SequenceLength]);
    /// <summary>A shaped read-only view when present; a default rank-zero view when absent. Check HasTokenTypeIds.</summary>
    public ReadOnlyTensorSpan<long> TokenTypeIds => TypeBuffer is null ? default : new(TypeBuffer, [BatchSize, SequenceLength]);
    public bool HasTokenTypeIds => TypeBuffer is not null;
}
