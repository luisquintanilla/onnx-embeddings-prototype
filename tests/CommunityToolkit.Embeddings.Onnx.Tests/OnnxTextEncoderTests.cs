using CommunityToolkit.Embeddings.Onnx;
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

public sealed class OnnxTextEncoderTests
{
    internal static TokenBatch Batch(bool types = true)
        => types ? new(1,2,[7,9],[1,0],[3,4]) : new(1,2,[7,9],[1,0]);

    [Theory]
    [InlineData("missing_ids", "input_ids")]
    [InlineData("missing_mask", "attention_mask")]
    [InlineData("unexpected", "position_ids")]
    [InlineData("float_ids", "Int64")]
    [InlineData("rank1_ids", "Int64")]
    [InlineData("rank3_ids", "Int64")]
    [InlineData("float_mask", "Int64")]
    [InlineData("rank1_mask", "Int64")]
    [InlineData("float_types", "Int64")]
    [InlineData("rank1_types", "Int64")]
    [InlineData("missing_output", "last_hidden_state")]
    [InlineData("pooled", "token tensor")]
    [InlineData("dimensions", "token tensor")]
    [InlineData("double_output", "Float32")]
    public void Constructor_RejectsInvalidGraphContract(string graph, string message)
    {
        using var session = TestAssets.Session(graph);
        var error = Assert.Throws<ArgumentException>(() => new OnnxTextEncoder(session, 3));
        Assert.Contains(message, error.Message);
    }

    [Fact]
    public void Constructor_ValidatesArgumentsAndNamedOutputWithoutTakingOwnershipOnFailure()
    {
        Assert.Throws<ArgumentNullException>(() => new OnnxTextEncoder(null!));
        using var session = TestAssets.Session("named_logits");
        Assert.Throws<ArgumentOutOfRangeException>(() => new OnnxTextEncoder(session, 0, "logits", true));
        Assert.Throws<ArgumentException>(() => new OnnxTextEncoder(session, 3, "", true));
        Assert.Throws<ArgumentException>(() => new OnnxTextEncoder(session, 3, ownsSession: true));
        using var encoder = new OnnxTextEncoder(session, 3, "logits");
        Assert.Equal(new float[] {7,1,1,9,0,1}, encoder.Score(Batch()).ToArray());
        Assert.Equal(3, encoder.Dimensions);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Score_OptionalTypesShapesCancellationAndManagedOwnership(bool required)
    {
        using var session = TestAssets.Session(required ? "required_types" : "optional_types");
        using var encoder = new OnnxTextEncoder(session, 3);
        Assert.Equal(required, encoder.RequiresTokenTypeIds);
        float[] expected = required ? [7,1,3,9,0,4] : [7,1,1,9,0,1];
        var first = encoder.Score(Batch());
        Assert.Equal(3, first.Rank);
        Assert.Equal(new nint[] {1,2,3}, first.Lengths.ToArray());
        Assert.Equal(expected, first.ToArray());
        if (required)
            Assert.Throws<ArgumentException>(() => encoder.Score(Batch(false)));
        else
            Assert.Equal(expected, encoder.Score(Batch(false)).ToArray());
        first[0,0,0] = 1234;
        var second = encoder.Score(Batch());
        Assert.Equal(expected, second.ToArray());
        Assert.NotSame(first, second);
        Assert.Empty(encoder.Score(new TokenBatch(0,0,[],[])).ToArray());
        Assert.Throws<ArgumentNullException>(() => encoder.Score(null!));
        using var cts = new CancellationTokenSource(); cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => encoder.Score(Batch(), cts.Token));
        Assert.Throws<OperationCanceledException>(() => encoder.Score(Batch(false), cts.Token));
        Assert.Throws<OperationCanceledException>(() => encoder.Score(new TokenBatch(0,0,[],[]), cts.Token));
        encoder.Dispose();
        session.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        Assert.Equal(expected, second.ToArray()); // native handles gone; owned shaped result remains valid
        Assert.Equal(1234, first[0,0,0]);
        Assert.Equal(expected[3], second[0,1,0]);
        Assert.Throws<ObjectDisposedException>(() => encoder.Score(Batch()));
    }

    [Fact]
    public void Score_EmptyTensorPreservesShapeUnlikePublicTensorEmpty()
    {
        Assert.Equal(0, Tensor<float>.Empty.Rank);
        Assert.Empty(Tensor<float>.Empty.Lengths.ToArray());
        Assert.Equal(0, Tensor<float>.Empty.FlattenedLength);
        var frameworkShapedEmpty = Tensor.CreateFromShape<float>([0,0,3]);
        Assert.Equal(3, frameworkShapedEmpty.Rank);
        Assert.Equal(new nint[] {0,0,3}, frameworkShapedEmpty.Lengths.ToArray());
        Assert.Equal(0, frameworkShapedEmpty.FlattenedLength);
        using var session = TestAssets.Session("required_types");
        using var encoder = new OnnxTextEncoder(session, 3);
        var output = encoder.Score(new TokenBatch(0,0,[],[]));
        Assert.Equal(3, output.Rank);
        Assert.Equal(new nint[] {0,0,3}, output.Lengths.ToArray());
        Assert.Equal(0, output.FlattenedLength);
        Assert.Empty(output.ToArray());
        // Empty inference does not demand optional inputs or prevent later real work.
        Assert.Equal(new float[] {7,1,3,9,0,4}, encoder.Score(Batch()).ToArray());
    }

    [Fact]
    public void Score_RejectsStaticBatchAndSequenceShapesAndRuntimeOutputDimensions()
    {
        using var fixedSession = TestAssets.Session("fixed_shape");
        using var fixedEncoder = new OnnxTextEncoder(fixedSession, 3);
        Assert.Throws<ArgumentException>(() => fixedEncoder.Score(new TokenBatch(1,3,[1,2,3],[1,1,1],[0,0,0])));
        Assert.Throws<ArgumentException>(() => fixedEncoder.Score(new TokenBatch(2,2,[1,2,3,4],[1,1,1,1],[0,0,0,0])));
        Assert.Equal(new float[] {1,1,0,2,1,0,3,0,0,4,1,0,5,1,0,6,1,0},
            fixedEncoder.Score(new TokenBatch(2,3,[1,2,3,4,5,6],[1,1,0,1,1,1],[0,0,0,0,0,0])).ToArray());
        foreach (string graph in new[] {"runtime_dimensions", "runtime_axis0", "runtime_axis1"})
        {
            using var runtimeSession = TestAssets.Session(graph);
            using var runtimeEncoder = new OnnxTextEncoder(runtimeSession, 3);
            var error = Assert.Throws<InvalidOperationException>(() => runtimeEncoder.Score(new TokenBatch(1,2,[1,2],[1,1])));
            Assert.Contains("[batch, sequence, 3]", error.Message);
        }
    }

    [Fact]
    public void Dispose_OwnedAndBorrowedSessionsBehaveDifferently()
    {
        using var borrowed = TestAssets.Session();
        var wrapper = new OnnxTextEncoder(borrowed, 3);
        wrapper.Dispose(); wrapper.Dispose();
        Assert.Throws<ObjectDisposedException>(() => wrapper.Score(Batch()));
        using var other = new OnnxTextEncoder(borrowed, 3);
        Assert.Equal(new float[] {7,1,1,9,0,1}, other.Score(Batch()).ToArray());
        using var owned = TestAssets.Session();
        var owner = new OnnxTextEncoder(owned, 3, ownsSession: true);
        owner.Dispose(); owner.Dispose();
        Assert.Throws<ObjectDisposedException>(() => owner.Score(Batch()));
        Assert.True(IsSessionDisposed(owned));
    }

    [Fact]
    public void Load_BorrowsOptionsAndValidatesPaths()
    {
        using var options = TestAssets.Options();
        var encoder = OnnxTextEncoder.Load(TestAssets.Graph("optional_types"), options, 3);
        var native = (InferenceSession)typeof(OnnxTextEncoder).GetField("_session",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(encoder)!;
        Assert.False(IsSessionDisposed(native));
        encoder.Dispose(); encoder.Dispose();
        Assert.True(IsSessionDisposed(native));
        Assert.False(options.IsClosed);
        using var second = OnnxTextEncoder.Load(TestAssets.Graph("required_types"), options, 3);
        Assert.Equal(new float[] {7,1,3,9,0,4}, second.Score(Batch()).ToArray());
        Assert.Throws<ArgumentException>(() => OnnxTextEncoder.Load("", options));
        Assert.Throws<FileNotFoundException>(() => OnnxTextEncoder.Load(Path.Combine(TestAssets.Root, ".assets", "not-a-model.onnx"), options));
        Assert.Throws<ArgumentException>(() => OnnxTextEncoder.Load(TestAssets.Graph("pooled"), options, 3));
        Assert.False(options.IsClosed);
        using var noOptions = OnnxTextEncoder.Load(TestAssets.Graph("optional_types"), dimensions: 3);
        Assert.Equal(new float[] {7,1,1,9,0,1}, noOptions.Score(Batch()).ToArray());
    }

    // ORT 1.23.2 exposes no safe public disposed probe; Run on a freed session can
    // access-violate. Inspect its managed lifetime flag ONLY for ownership assertions.
    internal static bool IsSessionDisposed(InferenceSession session)
        => (bool)typeof(InferenceSession).GetField("_disposed",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(session)!;
}
