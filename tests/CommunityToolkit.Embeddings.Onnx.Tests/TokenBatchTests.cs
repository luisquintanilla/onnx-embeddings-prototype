using CommunityToolkit.Embeddings.Onnx;
using System.Numerics.Tensors;

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

    [Fact]
    public void ReadOnlyViews_HaveShapeAndCopiedStorageWithoutMutablePublicAliases()
    {
        long[] ids = [91,7,8,9,10,92], mask = [91,1,0,1,1,92], types = [91,2,3,4,5,92];
        var batch = new TokenBatch(2, 2, ids.AsSpan(1,4), mask.AsSpan(1,4), types.AsSpan(1,4));
        ReadOnlyTensorSpan<long> idView = batch.InputIds;
        ReadOnlyTensorSpan<long> maskView = batch.AttentionMask;
        ReadOnlyTensorSpan<long> typeView = batch.TokenTypeIds;
        foreach (string name in new[] {nameof(TokenBatch.InputIds), nameof(TokenBatch.AttentionMask), nameof(TokenBatch.TokenTypeIds)})
        {
            var property = typeof(TokenBatch).GetProperty(name)!;
            Assert.Equal(typeof(ReadOnlyTensorSpan<long>), property.PropertyType);
            Assert.Null(property.SetMethod);
        }
        Assert.Empty(typeof(TokenBatch).GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance));
        Assert.Equal(2, idView.Rank);
        Assert.Equal(new nint[] {2,2}, idView.Lengths.ToArray());
        Assert.Equal(new nint[] {2,1}, idView.Strides.ToArray());
        Assert.Equal(new nint[] {2,2}, maskView.Lengths.ToArray());
        Assert.Equal(new nint[] {2,2}, typeView.Lengths.ToArray());
        Assert.True(idView.TryGetSpan(new nint[] {0,0}, 4, out ReadOnlySpan<long> copiedIds));
        Assert.False(ids.AsSpan().Overlaps(copiedIds));
        Array.Fill(ids, -1); Array.Fill(mask, -1); Array.Fill(types, -1);
        Assert.Equal(new long[] {7,8,9,10}, idView.ToArray());
        Assert.Equal(new long[] {1,0,1,1}, maskView.ToArray());
        Assert.Equal(new long[] {2,3,4,5}, typeView.ToArray());
        Assert.Equal(9, idView[1,0]);
        Assert.Equal(1, maskView[1,0]);
        Assert.Equal(4, typeView[1,0]);
        Assert.Equal(new long[] {7,8,9,10}, batch.InputIds.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EmptyViews_PreserveTwoAxesAndDistinguishAbsentTokenTypes(bool includeTypes)
    {
        var empty = TokenizerExtensionsTests.Tokenizer().PrepareBatch([], new(8, 101, 102, 1),
            includeTokenTypeIds: includeTypes);
        Assert.Equal(0, empty.BatchSize);
        Assert.Equal(0, empty.SequenceLength);
        Assert.Equal(2, empty.InputIds.Rank);
        Assert.Equal(new nint[] {0,0}, empty.InputIds.Lengths.ToArray());
        Assert.Equal(0, empty.InputIds.FlattenedLength);
        Assert.Equal(2, empty.AttentionMask.Rank);
        Assert.Equal(new nint[] {0,0}, empty.AttentionMask.Lengths.ToArray());
        Assert.Equal(0, empty.AttentionMask.FlattenedLength);
        Assert.Equal(includeTypes, empty.HasTokenTypeIds);
        Assert.Equal(includeTypes ? 2 : 0, empty.TokenTypeIds.Rank);
        Assert.Equal(includeTypes ? new nint[] {0,0} : [], empty.TokenTypeIds.Lengths.ToArray());
        Assert.Equal(0, empty.TokenTypeIds.FlattenedLength);

        var nonemptyWithoutTypes = new TokenBatch(1, 1, [7], [1]);
        Assert.False(nonemptyWithoutTypes.HasTokenTypeIds);
        Assert.Equal(0, nonemptyWithoutTypes.TokenTypeIds.Rank);
        Assert.Empty(nonemptyWithoutTypes.TokenTypeIds.Lengths.ToArray());
        Assert.Equal(0, nonemptyWithoutTypes.TokenTypeIds.FlattenedLength);
        Assert.Equal(new nint[] {1,1}, nonemptyWithoutTypes.InputIds.Lengths.ToArray());
        Assert.Equal(7, nonemptyWithoutTypes.InputIds[0,0]);
    }
}
