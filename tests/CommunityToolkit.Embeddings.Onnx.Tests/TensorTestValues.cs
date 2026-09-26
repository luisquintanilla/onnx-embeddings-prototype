using System.Numerics.Tensors;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

// Test-only snapshots for comparing public tensor values with independent arrays.
// Never used to prepare production inputs or to make a rejected layout contiguous.
internal static class TensorTestValues
{
    public static T[] ToArray<T>(this ReadOnlyTensorSpan<T> tensor)
    {
        T[] values = new T[checked((int)tensor.FlattenedLength)];
        if (values.Length != 0) tensor.FlattenTo(values);
        return values;
    }

    public static T[] ToArray<T>(this Tensor<T> tensor) => tensor.AsReadOnlyTensorSpan().ToArray();

    public static T[] Row<T>(this Tensor<T> tensor, int row)
    {
        Assert.Equal(2, tensor.Rank);
        T[] values = new T[checked((int)tensor.Lengths[1])];
        for (int column = 0; column < values.Length; column++) values[column] = tensor[row, column];
        return values;
    }
}
