using CommunityToolkit.Embeddings.Onnx;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

public sealed class EmbeddingPoolingTests
{
    [Fact]
    public void Pool_MeanAndClsUseCorrectTokensAndOwnBuffers()
    {
        var batch = new TokenBatch(2, 3, [1,2,0,3,4,5], [1,1,0,1,1,1]);
        float[] hidden = [2,4, 6,8, 1000,-1000, 1,3, 4,6, 7,9];
        var mean = EmbeddingPooling.Pool(new(hidden, [2,3,2]), batch.AttentionMask, PoolingMode.Mean, false);
        var cls = EmbeddingPooling.Pool(new(hidden, [2,3,2]), batch.AttentionMask, PoolingMode.Cls, false);
        Assert.Equal(new nint[] {2,2}, mean.Lengths.ToArray());
        Assert.Equal(new nint[] {2,2}, cls.Lengths.ToArray());
        Assert.Equal(new float[] {4,6}, mean.Row(0));
        Assert.Equal(new float[] {4,6}, mean.Row(1));
        Assert.Equal(new float[] {2,4}, cls.Row(0));
        Assert.Equal(new float[] {1,3}, cls.Row(1));
        mean[0,0] = 77;
        hidden[0] = 99;
        Assert.Equal(4, mean[1,0]);
        Assert.Equal(2, cls[0,0]);
        var normalized = EmbeddingPooling.Pool(new(new float[] {3,4}, [1,1,2]), new(new long[] {1}, [1,1]), PoolingMode.Mean);
        Assert.InRange(Math.Abs(normalized[0,0] - .6f), 0, 1e-7);
        Assert.InRange(Math.Abs(normalized[0,1] - .8f), 0, 1e-7);
    }

    [Theory]
    [InlineData(PoolingMode.Mean)]
    [InlineData(PoolingMode.Cls)]
    public void Pool_RejectsPaddingZeroAndNonfiniteEvenWhenMasked(PoolingMode mode)
    {
        Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.Pool(new(new float[] {3,4}, [1,1,2]), new(new long[] {0}, [1,1]), mode));
        Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.Pool(new(new float[] {0,0}, [1,1,2]), new(new long[] {1}, [1,1]), mode));
        foreach (float invalid in new[] {float.NaN, float.PositiveInfinity, float.NegativeInfinity})
            Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.Pool(new(new float[] {3,4,invalid,0}, [1,2,2]), new(new long[] {1,0}, [1,2]), mode));
        Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.Pool(new(new float[] {float.MaxValue,float.MaxValue}, [1,2,1]),
            new(new long[] {1,1}, [1,2]), PoolingMode.Mean, false));
    }

    [Fact]
    public void Pool_MeanAcceptsMaskedFirstTokenButClsRequiresFirstToken()
    {
        var batch = new TokenBatch(1,2,[0,7],[0,1]);
        Assert.Equal(new float[] {3,4}, EmbeddingPooling.Pool(new(new float[] {900,800,3,4}, [1,2,2]), batch.AttentionMask, PoolingMode.Mean, false).Row(0));
        Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.Pool(new(new float[] {900,800,3,4}, [1,2,2]), batch.AttentionMask, PoolingMode.Cls));
    }

    [Fact]
    public void Pool_ValidatesShapeModeDimensionsAndEmptyBatch()
    {
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(default, default, PoolingMode.Mean));
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(new(new float[] {1,2}, [1,1,2]), default, PoolingMode.Mean));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingPooling.Pool(new(new float[] {1}, [1,1,1]), new(new long[] {1}, [1,1]), (PoolingMode)42));
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(new(new float[] {1,2}, [1,2,1]), new(new long[] {1}, [1,1]), PoolingMode.Mean));
        var empty = EmbeddingPooling.Pool(Tensor.CreateFromShape<float>([0,0,2]),
            new(new long[0], [0,0]), PoolingMode.Mean);
        Assert.Equal(new nint[] {0,2}, empty.Lengths.ToArray());
        Assert.Empty(empty.ToArray());
    }

    [Theory]
    [InlineData(PoolingMode.Mean, false)]
    [InlineData(PoolingMode.Mean, true)]
    [InlineData(PoolingMode.Cls, false)]
    [InlineData(PoolingMode.Cls, true)]
    public void Pool_EmptyShapeIsPreservedAndPoolIntoDoesNotTouchBackingStorage(PoolingMode mode, bool normalize)
    {
        var hidden = Tensor.CreateFromShape<float>([0,0,3]);
        var attention = Tensor.CreateFromShape<long>([0,0]);
        var pooled = EmbeddingPooling.Pool(hidden, attention, mode, normalize);
        Assert.Equal(2, pooled.Rank);
        Assert.Equal(new nint[] {0,3}, pooled.Lengths.ToArray());
        Assert.Equal(0, pooled.FlattenedLength);
        float[] guards = [91,92];
        var destination = new TensorSpan<float>(guards, [0,3]);
        Assert.Equal(2, destination.Rank);
        EmbeddingPooling.PoolInto(hidden, attention, destination, mode, normalize);
        Assert.Equal(new float[] {91,92}, guards);
    }

    [Fact]
    public void Pool_EmptyInputsStillRequireRanksMatchingShapesAndPositiveHiddenDimension()
    {
        var hidden = Tensor.CreateFromShape<float>([0,0,3]);
        var attention = Tensor.CreateFromShape<long>([0,0]);
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(Tensor<float>.Empty, attention, PoolingMode.Mean));
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(hidden, Tensor<long>.Empty, PoolingMode.Mean));
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(
            Tensor.CreateFromShape<float>([0,0,0]), attention, PoolingMode.Mean));
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(
            Tensor.CreateFromShape<float>([1,0,3]), Tensor.CreateFromShape<long>([1,0]), PoolingMode.Mean));
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(hidden,
            Tensor.CreateFromShape<long>([0,1]), PoolingMode.Mean));
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.PoolInto(hidden, attention,
            Tensor.CreateFromShape<float>([0,4]).AsTensorSpan(), PoolingMode.Mean));
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.PoolInto(hidden, attention,
            Tensor<float>.Empty.AsTensorSpan(), PoolingMode.Mean));
    }

    [Theory]
    [InlineData("batch")]
    [InlineData("sequence")]
    [InlineData("hidden")]
    [InlineData("product")]
    public void Pool_RejectsDimensionAndProductOverflowBeforeMaterializingBuffers(string axis)
    {
        nint overInt = int.MaxValue;
        overInt = checked(overInt + 1);
        nint[] shape = axis switch
        {
            "batch" => [overInt,1,1],
            "sequence" => [1,overInt,1],
            "hidden" => [1,1,overInt],
            _ => [int.MaxValue,2,1]
        };
        // Broadcast metadata can represent the oversized shape over ONE scalar.
        // Construct first so an exception from pooling, not tensor construction,
        // is the evidence for its checked dimensions/product.
        var hidden = Tensor.Create(new float[] {1}, shape, [0,0,0]);
        var attention = Tensor.Create(new long[] {1}, [shape[0],shape[1]], [0,0]);
        Assert.Equal(shape, hidden.Lengths.ToArray());
        Assert.Throws<OverflowException>(() => EmbeddingPooling.Pool(hidden, attention, PoolingMode.Mean));
        Assert.Throws<OverflowException>(() => EmbeddingPooling.PoolInto(hidden, attention,
            new(new float[1], [1,1]), PoolingMode.Mean));
    }

    [Theory]
    [InlineData(PoolingMode.Mean, false)]
    [InlineData(PoolingMode.Mean, true)]
    [InlineData(PoolingMode.Cls, false)]
    [InlineData(PoolingMode.Cls, true)]
    public void PoolInto_OverwritesAndReusesDestination(PoolingMode mode, bool normalize)
    {
        float[] hidden = [3,4, 9,12, 0,5, 0,15];
        long[] mask = [1,1,1,0];
        float[] destination = [91,92,93,94];
        EmbeddingPooling.PoolInto(new(hidden, [2,2,2]), new(mask, [2,2]), new(destination, [2,2]), mode, normalize);
        float[] expected = normalize ? [.6f,.8f,0,1] : mode == PoolingMode.Mean ? [6,8,0,5] : [3,4,0,5];
        Assert.Equal(expected, destination);
        Assert.Equal(new float[] {3,4,9,12,0,5,0,15}, hidden);
        Assert.Equal(new long[] {1,1,1,0}, mask);
        // Different signs/values on the second call expose accumulation and stale rows.
        float[] next = [-3,4, -9,12, 5,0, 15,0];
        EmbeddingPooling.PoolInto(new(next, [2,2,2]), new(mask, [2,2]), new(destination, [2,2]), mode, normalize);
        float[] expectedNext = normalize ? [-.6f,.8f,1,0] : mode == PoolingMode.Mean ? [-6,8,5,0] : [-3,4,5,0];
        Assert.Equal(expectedNext, destination);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    public void PoolInto_RejectsOverlappingStorage(int offset)
    {
        float[] storage = [2,4,6,8,10,12];
        var hidden = new ReadOnlyTensorSpan<float>(storage.AsSpan(1,4), [1,2,2]);
        var destination = new TensorSpan<float>(storage.AsSpan(offset,2), [1,2]);
        Exception? error = null;
        try { EmbeddingPooling.PoolInto(hidden, new(new long[] {1,1}, [1,2]), destination, PoolingMode.Mean, false); }
        catch (Exception caught) { error = caught; }
        Assert.Equal("destination", Assert.IsType<ArgumentException>(error).ParamName);
        Assert.Equal(new float[] {2,4,6,8,10,12}, storage);
    }

    [Fact]
    public void PoolInto_AllowsDisjointViewsOfSameArray()
    {
        float[] storage = [2,4,6,8,91,92];
        var hidden = new ReadOnlyTensorSpan<float>(storage.AsSpan(0,4), [1,2,2]);
        var destination = new TensorSpan<float>(storage.AsSpan(4,2), [1,2]);
        EmbeddingPooling.PoolInto(hidden, new(new long[] {1,1}, [1,2]), destination, PoolingMode.Mean, false);
        Assert.Equal(new float[] {2,4,6,8,4,6}, storage);
    }

    [Fact]
    public void PoolInto_RejectsDestinationAliasingAttentionMaskBytes()
    {
        long[] attention = [1,1];
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.PoolInto(
            new(new float[] {3,4,6,8}, [1,2,2]), new(attention, [1,2]),
            new(MemoryMarshal.Cast<long,float>(attention.AsSpan()), [1,2]), PoolingMode.Mean, false));
        Assert.Equal(new long[] {1,1}, attention);
    }

    [Theory]
    [InlineData(PoolingMode.Mean)]
    [InlineData(PoolingMode.Cls)]
    public void Pool_UsesLogicalNonzeroOffsetSlicesNotArrayPrefix(PoolingMode mode)
    {
        float[] hidden = [float.NaN,float.NaN,float.NaN,float.NaN,float.NaN,float.NaN,
            2,4,6,8,1000,-1000, 1,3,4,6,7,9, float.PositiveInfinity];
        long[] attention = [-1,-1,-1, 1,1,0,1,1,1, 2];
        float[] output = [71,72,91,92,93,94,99];
        var hiddenView = new ReadOnlyTensorSpan<float>(hidden, [3,3,2]).Slice(new nint[] {1,0,0});
        var maskView = new ReadOnlyTensorSpan<long>(attention, [3,3]).Slice(new nint[] {1,0});
        var destination = new TensorSpan<float>(output, [3,2]).Slice(new nint[] {1,0});
        Assert.Equal(new nint[] {2,3,2}, hiddenView.Lengths.ToArray());
        Assert.Equal(new nint[] {2,3}, maskView.Lengths.ToArray());
        Assert.Equal(new nint[] {2,2}, destination.Lengths.ToArray());
        Assert.True(hiddenView.TryGetSpan(new nint[] {0,0,0}, 12, out var logicalHidden));
        Assert.Equal(new float[] {2,4,6,8,1000,-1000,1,3,4,6,7,9}, logicalHidden.ToArray());
        float[] expected = mode == PoolingMode.Mean ? [4,6,4,6] : [2,4,1,3];
        Assert.Equal(expected, EmbeddingPooling.Pool(hiddenView, maskView, mode, false).ToArray());
        EmbeddingPooling.PoolInto(hiddenView, maskView, destination, mode, false);
        Assert.Equal(new float[] {71,72,expected[0],expected[1],expected[2],expected[3],99}, output);
        Assert.True(float.IsNaN(hidden[0]));
        Assert.Equal(-1, attention[0]);
        Assert.Equal(2, attention[^1]);
    }

    [Theory]
    [InlineData("hidden", "permutation")]
    [InlineData("hidden", "strided")]
    [InlineData("hidden", "broadcast")]
    [InlineData("attention", "permutation")]
    [InlineData("attention", "strided")]
    [InlineData("attention", "broadcast")]
    [InlineData("destination", "permutation")]
    [InlineData("destination", "strided")]
    [InlineData("destination", "broadcast")]
    public void Pool_RejectsNonRowMajorLayoutsIncludingPermutations(string target, string layout)
    {
        float[] hiddenValues = Enumerable.Range(1, 16).Select(value => (float)value).ToArray();
        long[] maskValues = Enumerable.Repeat(1L, 8).ToArray();
        float[] outputValues = Enumerable.Repeat(91f, 8).ToArray();
        nint[] hiddenStrides = target != "hidden" ? [4,2,1] : layout switch
        {
            "permutation" => [4,1,2], "strided" => [8,4,2], _ => [0,2,1]
        };
        nint[] maskStrides = target != "attention" ? [2,1] : layout switch
        {
            "permutation" => [1,2], "strided" => [4,2], _ => [0,1]
        };
        nint[] outputStrides = target != "destination" ? [2,1] : layout switch
        {
            "permutation" => [1,2], "strided" => [4,2], _ => [0,1]
        };
        // Construct OUTSIDE the exception assertion: layout must be rejected by
        // pooling, not merely by the framework's tensor constructor.
        var hidden = Tensor.Create(hiddenValues, [2,2,2], hiddenStrides);
        var mask = Tensor.Create(maskValues, [2,2], maskStrides);
        var destination = Tensor.Create(outputValues, [2,2], outputStrides);
        if (layout == "permutation")
        {
            // Pinned Tensor10 defines IsDense in logical traversal order: a
            // gapless physical permutation is NOT IsDense. Do not assume it is.
            Assert.Equal(target != "hidden", hidden.IsDense);
            Assert.Equal(target != "attention", mask.IsDense);
            Assert.Equal(target != "destination", destination.IsDense);
        }
        if (target != "destination")
            Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(hidden, mask, PoolingMode.Mean, false));
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.PoolInto(hidden, mask, destination.AsTensorSpan(), PoolingMode.Mean, false));
        Assert.Equal(Enumerable.Repeat(91f, 8), outputValues);
    }

    [Fact]
    public void Pool_RejectsInvalidRanksShapesAndDestinationDimensions()
    {
        var validHidden = Tensor.Create(new float[] {2,4,6,8}, [1,2,2]);
        var validMask = Tensor.Create(new long[] {1,1}, [1,2]);
        foreach (nint[] shape in new nint[][] {[4], [2,2], [1,1,2,2], [2,1,2]})
        {
            var invalidHidden = Tensor.Create(new float[] {2,4,6,8}, shape);
            Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(invalidHidden, validMask, PoolingMode.Mean));
            Assert.Throws<ArgumentException>(() => EmbeddingPooling.PoolInto(invalidHidden, validMask,
                new(new float[2], [1,2]), PoolingMode.Mean));
        }
        foreach (nint[] shape in new nint[][] {[2], [1,1,2], [2,1]})
        {
            var invalidMask = Tensor.Create(new long[] {1,1}, shape);
            Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(validHidden, invalidMask, PoolingMode.Mean));
            Assert.Throws<ArgumentException>(() => EmbeddingPooling.PoolInto(validHidden, invalidMask,
                new(new float[2], [1,2]), PoolingMode.Mean));
        }
        foreach (nint[] shape in new nint[][] {[2], [1,1,2], [2,1], [1,1], [1,3]})
        {
            var destination = Tensor.CreateFromShape<float>(shape);
            Assert.Throws<ArgumentException>(() => EmbeddingPooling.PoolInto(validHidden, validMask,
                destination.AsTensorSpan(), PoolingMode.Mean));
        }
        Assert.Throws<ArgumentException>(() => EmbeddingPooling.PoolInto(validHidden, validMask, default, PoolingMode.Mean));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingPooling.PoolInto(validHidden, validMask,
            new(new float[2], [1,2]), (PoolingMode)99));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(2)]
    [InlineData(long.MaxValue)]
    public void Pool_DirectAttentionViewRejectsNonBinaryValues(long invalid)
    {
        var hidden = Tensor.Create(new float[] {3,4,6,8}, [1,2,2]);
        foreach (PoolingMode mode in new[] {PoolingMode.Mean, PoolingMode.Cls})
        foreach (long[] values in new long[][] {[invalid,1], [1,invalid]})
        {
            var mask = Tensor.Create(values, [1,2]);
            Assert.Throws<ArgumentException>(() => EmbeddingPooling.Pool(hidden, mask, mode, false));
            float[] destination = [91,92];
            Assert.Throws<ArgumentException>(() => EmbeddingPooling.PoolInto(hidden, mask, new(destination, [1,2]), mode, false));
            Assert.Equal(new float[] {91,92}, destination);
        }
    }

    [Theory]
    [InlineData(PoolingMode.Mean)]
    [InlineData(PoolingMode.Cls)]
    public void PoolInto_RejectsInvalidNumericsIncludingMaskedStates(PoolingMode mode)
    {
        foreach (float invalid in new[] {float.NaN,float.PositiveInfinity,float.NegativeInfinity})
        foreach (bool normalize in new[] {false,true})
        {
            float[] destination = [91,92];
            Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.PoolInto(
                new(new float[] {3,4,invalid,0}, [1,2,2]), new(new long[] {1,0}, [1,2]),
                new(destination, [1,2]), mode, normalize));
            Assert.Equal(new float[] {91,92}, destination);
        }
        Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.PoolInto(
            new(new float[] {3,4}, [1,1,2]), new(new long[] {0}, [1,1]), new(new float[2], [1,2]), mode));
        Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.PoolInto(
            new(new float[] {0,0}, [1,1,2]), new(new long[] {1}, [1,1]), new(new float[2], [1,2]), mode));
        float[] unnormalizedZero = [91,92];
        EmbeddingPooling.PoolInto(new(new float[] {0,0}, [1,1,2]), new(new long[] {1}, [1,1]),
            new(unnormalizedZero, [1,2]), mode, false);
        Assert.Equal(new float[] {0,0}, unnormalizedZero);
    }

    [Theory]
    [InlineData(PoolingMode.Mean, false, 4f)]
    [InlineData(PoolingMode.Cls, false, 2f)]
    [InlineData(PoolingMode.Mean, true, 1f)]
    [InlineData(PoolingMode.Cls, true, 1f)]
    public void PoolInto_AcceptsSingletonHiddenDimension(PoolingMode mode, bool normalize, float expected)
    {
        float[] input = [2,6], destination = [91];
        var hidden = new ReadOnlyTensorSpan<float>(input, [1,2,1]);
        Assert.Equal(new nint[] {1,2,1}, hidden.Lengths.ToArray());
        EmbeddingPooling.PoolInto(hidden, new(new long[] {1,1}, [1,2]), new(destination, [1,1]), mode, normalize);
        Assert.Equal(new float[] {expected}, destination);
        Assert.Equal(new float[] {2,6}, input);
        var allocated = EmbeddingPooling.Pool(hidden, new(new long[] {1,1}, [1,2]), mode, normalize);
        Assert.Equal(new nint[] {1,1}, allocated.Lengths.ToArray());
        Assert.Equal(expected, allocated[0,0]);
    }

    [Theory]
    [InlineData(PoolingMode.Mean, false)]
    [InlineData(PoolingMode.Mean, true)]
    [InlineData(PoolingMode.Cls, false)]
    [InlineData(PoolingMode.Cls, true)]
    public void PoolInto_SingleElementAndSqueezedOffsetViewsUseLogicalValues(PoolingMode mode, bool normalize)
    {
        float[] scalarInput = [float.NaN,3,float.PositiveInfinity], scalarOutput = [91,-9,92];
        long[] scalarMask = [-1,1,2];
        var scalarHidden = new ReadOnlyTensorSpan<float>(scalarInput.AsSpan(1,1), [1,1,1]);
        var scalarAttention = new ReadOnlyTensorSpan<long>(scalarMask.AsSpan(1,1), [1,1]);
        var scalarDestination = new TensorSpan<float>(scalarOutput.AsSpan(1,1), [1,1]);
        EmbeddingPooling.PoolInto(scalarHidden, scalarAttention, scalarDestination, mode, normalize);
        float expectedScalar = normalize ? 1 : 3;
        Assert.Equal(new float[] {91,expectedScalar,92}, scalarOutput);
        Assert.Equal(new float[] {expectedScalar}, EmbeddingPooling.Pool(scalarHidden, scalarAttention, mode, normalize).ToArray());
        Assert.Equal(new long[] {-1,1,2}, scalarMask);

        // All three views now have >1 element with trailing singleton axes;
        // this exercises the metadata-only Squeeze path rather than the scalar path.
        float[] rowsInput = [float.NaN,2,6,float.PositiveInfinity], rowsOutput = [91,-9,-9,92];
        long[] rowsMask = [-1,1,1,2];
        var rowsHidden = new ReadOnlyTensorSpan<float>(rowsInput.AsSpan(1,2), [2,1,1]);
        var rowsAttention = new ReadOnlyTensorSpan<long>(rowsMask.AsSpan(1,2), [2,1]);
        var rowsDestination = new TensorSpan<float>(rowsOutput.AsSpan(1,2), [2,1]);
        EmbeddingPooling.PoolInto(rowsHidden, rowsAttention, rowsDestination, mode, normalize);
        float[] expectedRows = normalize ? [1,1] : [2,6];
        Assert.Equal(new float[] {91,expectedRows[0],expectedRows[1],92}, rowsOutput);
        Assert.Equal(expectedRows, EmbeddingPooling.Pool(rowsHidden, rowsAttention, mode, normalize).ToArray());
        Assert.Equal(new long[] {-1,1,1,2}, rowsMask);
        Assert.Equal(2, rowsInput[1]);
        Assert.Equal(6, rowsInput[2]);
    }

    [Fact]
    public void PoolInto_OffsetMultirowSequenceWithSingletonHiddenUsesLogicalValuesAndPreservesGuards()
    {
        float[] input = [float.NaN,float.NaN, 2,6,10,9000, float.PositiveInfinity];
        long[] masks = [2,2, 1,1,1,0, -1];
        float[] output = [91,-9,-9,92];
        var hidden = new ReadOnlyTensorSpan<float>(input, [3,2,1]).Slice(new nint[] {1,0,0});
        var attention = new ReadOnlyTensorSpan<long>(masks, [3,2]).Slice(new nint[] {1,0});
        var destination = new TensorSpan<float>(output.AsSpan(1,2), [2,1]);
        Assert.Equal(new nint[] {2,2,1}, hidden.Lengths.ToArray());
        Assert.Equal(new nint[] {2,2}, attention.Lengths.ToArray());
        Assert.Equal(new nint[] {2,1}, destination.Lengths.ToArray());
        Assert.Equal(0, hidden.Strides[2]);
        Assert.Equal(0, destination.Strides[1]);

        EmbeddingPooling.PoolInto(hidden, attention, destination, PoolingMode.Mean, normalize: false);

        // Row0 averages 2 and6; row1 uses10 and excludes the masked9000.
        Assert.Equal(new float[] {91,4,10,92}, output);
        Assert.Equal(new float[] {2,6,10,9000}, input.AsSpan(2,4).ToArray());
        Assert.True(float.IsNaN(input[0]));
        Assert.True(float.IsPositiveInfinity(input[^1]));
        Assert.Equal(new long[] {2,2,1,1,1,0,-1}, masks);
        var allocated = EmbeddingPooling.Pool(hidden, attention, PoolingMode.Mean, normalize: false);
        Assert.Equal(new nint[] {2,1}, allocated.Lengths.ToArray());
        Assert.Equal(new float[] {4,10}, allocated.ToArray());
    }

    [Fact]
    public void PoolInto_RejectsAccumulationOverflowWithAndWithoutNormalization()
    {
        foreach (bool normalize in new[] {false,true})
        {
            Assert.Throws<InvalidOperationException>(() => EmbeddingPooling.PoolInto(
                new(new float[] {float.MaxValue,float.MaxValue}, [1,2,1]), new(new long[] {1,1}, [1,2]),
                new(new float[1], [1,1]), PoolingMode.Mean, normalize));
        }
    }

    [Theory]
    [InlineData(3e30f, -4e30f)]
    [InlineData(3e-30f, -4e-30f)]
    [InlineData(3e-44f, -4e-44f)]
    [InlineData(float.MaxValue, float.MinValue)]
    public void PoolInto_NormalizesExtremeMagnitudesUsingDestinationStorage(float a, float b)
    {
        float[] input = [a,b], output = [91,92];
        double norm = Math.Sqrt((double)a * a + (double)b * b);
        foreach (PoolingMode mode in new[] {PoolingMode.Mean, PoolingMode.Cls})
        {
            EmbeddingPooling.PoolInto(new(input, [1,1,2]), new(new long[] {1}, [1,1]),
                new(output, [1,2]), mode);
            Assert.InRange(Math.Abs(output[0] - a / norm), 0, 1e-6);
            Assert.InRange(Math.Abs(output[1] - b / norm), 0, 1e-6);
            Assert.InRange(Math.Abs(output.Sum(value => (double)value * value) - 1), 0, 2e-6);
            Assert.Equal(new float[] {a,b}, input);
        }
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
