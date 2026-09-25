using CommunityToolkit.Embeddings.Onnx;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

public sealed class TokenBatchTests
{
    [Fact]
    public void Constructor_CopiesAndValidatesBuffers()
    {
        long[] ids = [7, 8, 9, 10], mask = [1, 1, 1, 0], types = [0, 1, 2, 3];
        var batch = new TokenBatch(2, 2, ids, mask, types);
        Array.Fill(ids, 99); Array.Fill(mask, 0); Array.Fill(types, 99);
        Assert.Equal(2, batch.BatchSize);
        Assert.Equal(2, batch.SequenceLength);
        Assert.Equal(new long[] {7,8,9,10}, batch.InputIds.ToArray());
        Assert.Equal(new long[] {1,1,1,0}, batch.AttentionMask.ToArray());
        Assert.Equal(new long[] {0,1,2,3}, batch.TokenTypeIds.ToArray());
        Assert.True(batch.HasTokenTypeIds);
        var noTypes = new TokenBatch(1, 1, [1], [1]);
        Assert.False(noTypes.HasTokenTypeIds);
        Assert.Empty(noTypes.TokenTypeIds.ToArray());
        var empty = new TokenBatch(0, 0, [], []);
        Assert.Equal(0, empty.SequenceLength);
        Assert.Empty(empty.InputIds.ToArray());
    }

    [Theory]
    [InlineData(-1, 2, "negative")]
    [InlineData(1, -1, "negative")]
    [InlineData(0, 2, "empty")]
    [InlineData(2, 0, "empty")]
    [InlineData(2, 2, "ids")]
    [InlineData(1, 1, "mask-length")]
    [InlineData(1, 1, "types-length")]
    [InlineData(1, 1, "negative-id")]
    [InlineData(1, 1, "negative-type")]
    [InlineData(1, 1, "mask-negative")]
    [InlineData(1, 1, "mask-two")]
    public void Constructor_RejectsInvalidDimensionsIdsMasksAndTypes(int rows, int length, string fault)
    {
        long[] ids = [fault == "negative-id" ? -1 : 7];
        long[] mask = fault == "mask-length" ? [] : [fault == "mask-two" ? 2 : fault == "mask-negative" ? -1 : 1];
        long[] types = fault == "types-length" ? [0, 1] : [fault == "negative-type" ? -1 : 0];
        Assert.ThrowsAny<ArgumentException>(() => new TokenBatch(rows, length, ids, mask, types));
    }

    [Fact]
    public void Constructor_RejectsShapeProductOverflow()
        => Assert.Throws<OverflowException>(() => new TokenBatch(int.MaxValue, 2, [], []));
}
