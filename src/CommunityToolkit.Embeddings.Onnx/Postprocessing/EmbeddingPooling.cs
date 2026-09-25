using System.Numerics.Tensors;

namespace CommunityToolkit.Embeddings.Onnx;

public enum PoolingMode { Mean, Cls }

/// <summary>Pooling of explicit token-level [batch, sequence, hidden] Float32 outputs.</summary>
public static class EmbeddingPooling
{
    public static float[][] Pool(ReadOnlySpan<float> hiddenStates, TokenBatch batch, int dimensions,
        PoolingMode mode, bool normalize = true)
    {
        ArgumentNullException.ThrowIfNull(batch);
        ArgumentOutOfRangeException.ThrowIfLessThan(dimensions, 1);
        if (!Enum.IsDefined(mode)) throw new ArgumentOutOfRangeException(nameof(mode));
        if (hiddenStates.Length != checked(batch.BatchSize * batch.SequenceLength * dimensions))
            throw new ArgumentException("Hidden states must have shape [batch, sequence, dimensions].", nameof(hiddenStates));
        // Reject nonfinite values even in masked positions; do not hide a broken model output.
        foreach (float value in hiddenStates)
            if (!float.IsFinite(value)) throw new InvalidOperationException("Model output contains a nonfinite value.");

        var result = new float[batch.BatchSize][];
        var mask = batch.AttentionMask.Span;
        for (int row = 0; row < batch.BatchSize; row++)
        {
            float[] vector = new float[dimensions];
            int start = row * batch.SequenceLength;
            if (mode == PoolingMode.Cls)
            {
                if (mask[start] != 1) throw new InvalidOperationException("CLS pooling requires an unmasked first token.");
                hiddenStates.Slice(start * dimensions, dimensions).CopyTo(vector);
            }
            else
            {
                int tokens = 0;
                for (int token = 0; token < batch.SequenceLength; token++)
                {
                    if (mask[start + token] == 0) continue;
                    tokens++;
                    TensorPrimitives.Add<float>(vector, hiddenStates.Slice((start + token) * dimensions, dimensions), vector);
                }
                if (tokens == 0) throw new InvalidOperationException("Mean pooling cannot process an all-padding row.");
                TensorPrimitives.Divide<float>(vector, tokens, vector);
            }
            if (normalize) Normalize(vector);
            else
                foreach (float value in vector)
                    if (!float.IsFinite(value)) throw new InvalidOperationException("Pooling overflowed.");
            result[row] = vector;
        }
        return result;
    }

    /// <summary>In-place L2 normalization. Zero and nonfinite vectors are errors, not valid embeddings.</summary>
    public static void Normalize(Span<float> vector)
    {
        float maximum = 0;
        foreach (float value in vector)
        {
            if (!float.IsFinite(value)) throw new InvalidOperationException("Cannot normalize a nonfinite vector.");
            maximum = Math.Max(maximum, Math.Abs(value));
        }
        if (maximum == 0) throw new InvalidOperationException("Cannot normalize a zero vector.");
        // Scaling first avoids squared-norm overflow and underflow, including subnormal inputs.
        TensorPrimitives.Divide<float>(vector, maximum, vector);
        float norm = TensorPrimitives.Norm<float>(vector);
        TensorPrimitives.Divide<float>(vector, norm, vector);
    }
}
