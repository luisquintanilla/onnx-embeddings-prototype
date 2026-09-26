using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace CommunityToolkit.Embeddings.Onnx;

public enum PoolingMode { Mean, Cls }

/// <summary>Pooling of explicit token-level [batch, sequence, hidden] Float32 outputs.</summary>
public static class EmbeddingPooling
{
    public static Tensor<float> Pool(ReadOnlyTensorSpan<float> hiddenStates, ReadOnlyTensorSpan<long> attentionMask,
        PoolingMode mode, bool normalize = true)
    {
        var (batch, _, dimensions) = ValidateShape(hiddenStates, attentionMask, mode);
        var result = Tensor.CreateFromShape<float>([batch, dimensions]);
        PoolInto(hiddenStates, attentionMask, result.AsTensorSpan(), mode, normalize);
        return result;
    }

    /// <summary>Writes [batch, hidden] into caller storage. All tensors must be contiguous row-major views; input/output overlap is rejected.</summary>
    /// <remarks>Nonzero-offset contiguous slices are supported. On numerical failure the destination may be partially written.</remarks>
    public static void PoolInto(ReadOnlyTensorSpan<float> hiddenStates, ReadOnlyTensorSpan<long> attentionMask,
        TensorSpan<float> destination, PoolingMode mode, bool normalize = true)
    {
        var (batch, sequence, dimensions) = ValidateShape(hiddenStates, attentionMask, mode);
        if (destination.Rank != 2 || destination.Lengths[0] != batch || destination.Lengths[1] != dimensions)
            throw new ArgumentException("Destination must have shape [batch, hidden].", nameof(destination));
        ValidateLayout(destination.Lengths, destination.Strides, nameof(destination));
        var hidden = ContiguousValues(hiddenStates);
        var mask = ContiguousValues(attentionMask);
        var output = ContiguousValues(destination);
        if (hidden.Overlaps(output) || MemoryMarshal.AsBytes(mask).Overlaps(MemoryMarshal.AsBytes(output)))
            throw new ArgumentException("Destination must not overlap hidden states or attention masks.", nameof(destination));
        foreach (long value in mask)
            if (value is not (0 or 1)) throw new ArgumentException("Attention masks must contain only zero or one.", nameof(attentionMask));
        // Reject nonfinite values even in masked positions; do not hide a broken model output.
        foreach (float value in hidden)
            if (!float.IsFinite(value)) throw new InvalidOperationException("Model output contains a nonfinite value.");

        for (int row = 0; row < batch; row++)
        {
            Span<float> vector = output.Slice(row * dimensions, dimensions);
            vector.Clear();
            int start = row * sequence;
            if (mode == PoolingMode.Cls)
            {
                if (mask[start] != 1) throw new InvalidOperationException("CLS pooling requires an unmasked first token.");
                hidden.Slice(start * dimensions, dimensions).CopyTo(vector);
            }
            else
            {
                int tokens = 0;
                for (int token = 0; token < sequence; token++)
                {
                    if (mask[start + token] == 0) continue;
                    tokens++;
                    TensorPrimitives.Add<float>(vector, hidden.Slice((start + token) * dimensions, dimensions), vector);
                }
                if (tokens == 0) throw new InvalidOperationException("Mean pooling cannot process an all-padding row.");
                TensorPrimitives.Divide<float>(vector, tokens, vector);
            }
            if (normalize) Normalize(vector);
            else
                foreach (float value in vector)
                    if (!float.IsFinite(value)) throw new InvalidOperationException("Pooling overflowed.");
        }
    }

    private static (int Batch, int Sequence, int Hidden) ValidateShape(
        ReadOnlyTensorSpan<float> hidden, ReadOnlyTensorSpan<long> mask, PoolingMode mode)
    {
        if (!Enum.IsDefined(mode)) throw new ArgumentOutOfRangeException(nameof(mode));
        if (hidden.Rank != 3 || mask.Rank != 2 || hidden.Lengths[0] != mask.Lengths[0] || hidden.Lengths[1] != mask.Lengths[1])
            throw new ArgumentException("Hidden states [batch, sequence, hidden] and attention mask [batch, sequence] must agree.");
        int batch = checked((int)hidden.Lengths[0]);
        int sequence = checked((int)hidden.Lengths[1]);
        int dimensions = checked((int)hidden.Lengths[2]);
        if (dimensions < 1 || (batch > 0 && sequence < 1))
            throw new ArgumentException("Hidden dimension must be positive and nonempty batches require tokens.");
        _ = checked(batch * sequence * dimensions);
        ValidateLayout(hidden.Lengths, hidden.Strides, nameof(hidden));
        ValidateLayout(mask.Lengths, mask.Strides, nameof(mask));
        return (batch, sequence, dimensions);
    }

    private static void ValidateLayout(ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides, string name)
    {
        if (lengths.Contains((nint)0)) return;
        nint expected = 1;
        for (int axis = lengths.Length - 1; axis >= 0; axis--)
        {
            if (lengths[axis] > 1 && strides[axis] != expected)
                throw new ArgumentException("Only contiguous row-major tensor layouts are supported; materialize other layouts explicitly.", name);
            expected = checked(expected * lengths[axis]);
        }
    }

    // The pinned GetSpan misclassifies trailing singleton axes (stride zero).
    // Squeeze changes only view metadata; explicit layout validation has already rejected broadcasts.
    private static ReadOnlySpan<T> ContiguousValues<T>(ReadOnlyTensorSpan<T> tensor)
    {
        if (tensor.IsEmpty) return [];
        Span<nint> start = stackalloc nint[tensor.Rank];
        start.Clear();
        if (tensor.FlattenedLength == 1) return new ReadOnlySpan<T>(in tensor[start]);
        if (tensor.Lengths.Contains((nint)1)) tensor = tensor.Squeeze();
        return tensor.GetSpan(start[..tensor.Rank], checked((int)tensor.FlattenedLength));
    }

    private static Span<T> ContiguousValues<T>(TensorSpan<T> tensor)
    {
        if (tensor.IsEmpty) return [];
        Span<nint> start = stackalloc nint[tensor.Rank];
        start.Clear();
        if (tensor.FlattenedLength == 1) return new Span<T>(ref tensor[start]);
        if (tensor.Lengths.Contains((nint)1)) tensor = tensor.Squeeze();
        return tensor.GetSpan(start[..tensor.Rank], checked((int)tensor.FlattenedLength));
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
