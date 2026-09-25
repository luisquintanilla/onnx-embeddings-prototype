using CommunityToolkit.Embeddings.Onnx;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

public sealed class EmbeddingPoolingTests
{
    [Fact]
    public void Pool_MeanAndClsUseCorrectTokensAndOwnBuffers()
    {
        var batch = new TokenBatch(2, 3, [1,2,0,3,4,5], [1,1,0,1,1,1]);
        float[] hidden = [2,4, 6,8, 1000,-1000, 1,3, 4,6, 7,9];
        var mean = EmbeddingPooling.Pool(hidden, batch, 2, PoolingMode.Mean, false);
        var cls = EmbeddingPooling.Pool(hidden, batch, 2, PoolingMode.Cls, false);
        Assert.Equal(new float[] {4,6}, mean[0]);
        Assert.Equal(new float[] {4,6}, mean[1]);
        Assert.Equal(new float[] {2,4}, cls[0]);
        Assert.Equal(new float[] {1,3}, cls[1]);
        mean[0][0] = 77;
        hidden[0] = 99;
        Assert.Equal(4, mean[1][0]);
        Assert.Equal(2, cls[0][0]);
        var normalized = EmbeddingPooling.Pool([3,4], new TokenBatch(1,1,[1],[1]), 2, PoolingMode.Mean);
        Assert.InRange(Math.Abs(normalized[0][0] - .6f), 0, 1e-7);
        Assert.InRange(Math.Abs(normalized[0][1] - .8f), 0, 1e-7);
    }

    [Theory]
    [InlineData(PoolingMode.Mean)]
    [InlineData(PoolingMode.Cls)]
    public void Pool_RejectsPaddingZeroAndNonfiniteEvenWhenMasked(PoolingMode mode)
    {
        Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.Pool([3,4], new TokenBatch(1,1,[0],[0]), 2, mode));
        Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.Pool([0,0], new TokenBatch(1,1,[1],[1]), 2, mode));
        foreach (float invalid in new[] {float.NaN, float.PositiveInfinity, float.NegativeInfinity})
            Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.Pool([3,4,invalid,0], new TokenBatch(1,2,[1,0],[1,0]), 2, mode));
        Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.Pool([float.MaxValue,float.MaxValue],
            new TokenBatch(1,2,[1,1],[1,1]), 1, PoolingMode.Mean, false));
    }

    [Fact]
    public void Pool_MeanAcceptsMaskedFirstTokenButClsRequiresFirstToken()
    {
        var batch = new TokenBatch(1,2,[0,7],[0,1]);
        Assert.Equal(new float[] {3,4}, Assert.Single(EmbeddingPooling.Pool([900,800,3,4], batch, 2, PoolingMode.Mean, false)));
        Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.Pool([900,800,3,4], batch, 2, PoolingMode.Cls));
    }

    [Fact]
    public void Pool_ValidatesShapeModeDimensionsAndEmptyBatch()
    {
        var batch = new TokenBatch(1,1,[1],[1]);
        Assert.Throws<ArgumentNullException>(() => EmbeddingPooling.Pool([], null!, 2, PoolingMode.Mean));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingPooling.Pool([], batch, 0, PoolingMode.Mean));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingPooling.Pool([1], batch, 1, (PoolingMode)42));
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool([1,2], batch, 1, PoolingMode.Mean));
        Assert.Empty(EmbeddingPooling.Pool([], new TokenBatch(0,0,[],[]), 2, PoolingMode.Mean));
    }

    [Theory]
    [InlineData(3e30f, -4e30f)]
    [InlineData(3e-30f, -4e-30f)]
    [InlineData(3e-44f, -4e-44f)]
    [InlineData(float.MaxValue, float.MinValue)]
    public void Normalize_ExtremeMagnitudesAreStable(float a, float b)
    {
        float[] vector = [a,b];
        double norm = Math.Sqrt((double)a * a + (double)b * b);
        EmbeddingPooling.Normalize(vector);
        Assert.InRange(Math.Abs(vector[0] - a / norm), 0, 1e-6);
        Assert.InRange(Math.Abs(vector[1] - b / norm), 0, 1e-6);
        Assert.InRange(Math.Abs(vector.Sum(x => (double)x * x) - 1), 0, 2e-6);
    }

    [Fact]
    public void Normalize_RejectsZeroEmptyAndNonfinite()
    {
        foreach (float[] vector in new float[][] {[], [0,0], [float.NaN,1], [1,float.PositiveInfinity], [float.NegativeInfinity]})
            Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.Normalize(vector));
    }
}
