using CommunityToolkit.Embeddings.Onnx;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

public sealed class E5TextTests
{
    [Theory]
    [InlineData("", E5Purpose.Query, "query: ")]
    [InlineData("  Café \t", E5Purpose.Document, "passage:   Café \t")]
    [InlineData("query: cats", E5Purpose.Query, "query: query: cats")]
    [InlineData("passage: cats", E5Purpose.Document, "passage: passage: cats")]
    [InlineData("passage: cats", E5Purpose.Query, "query: passage: cats")]
    public void Format_PreservesCallerTextAndAlwaysAddsSelectedPrefix(string text, E5Purpose purpose, string expected)
        => Assert.Equal(expected, E5Text.Format(text, purpose));

    [Fact]
    public void Format_RejectsNullAndUnknownPurpose()
    {
        Assert.Throws<ArgumentNullException>(() => E5Text.Format(null!, E5Purpose.Query));
        Assert.Throws<ArgumentOutOfRangeException>(() => E5Text.Format("x", (E5Purpose)2));
    }
}
