using CommunityToolkit.Embeddings.Onnx;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

public sealed class TextBatchPreparerTests
{
    internal static TextTokenizer Tokenizer(Func<string, IReadOnlyList<int>>? encode = null, int pad = 9)
        => new(encode ?? (text => text.Select(c => (int)c).ToArray()), 101, 102, pad);

    [Theory]
    [InlineData(0)]
    [InlineData(9)]
    public void Prepare_BudgetsSpecialsTruncatesAndPadsInOrder(int pad)
    {
        var preparer = new TextBatchPreparer(Tokenizer(pad: pad), 5, 4);
        var batch = preparer.Prepare(["a", "bc", "xyz", "ghij"]);
        Assert.Equal(4, batch.BatchSize);
        Assert.Equal(5, batch.SequenceLength);
        Assert.Equal(new long[] {101,97,102,pad,pad, 101,98,99,102,pad, 101,120,121,122,102, 101,103,104,105,102}, batch.InputIds.ToArray());
        Assert.Equal(new long[] {1,1,1,0,0, 1,1,1,1,0, 1,1,1,1,1, 1,1,1,1,1}, batch.AttentionMask.ToArray());
        Assert.True(batch.HasTokenTypeIds);
        Assert.Equal(new long[20], batch.TokenTypeIds.ToArray());
        Assert.Throws<ArgumentException>(() => preparer.Prepare(["a","b","c","d","e"]));
    }

    [Fact]
    public void Prepare_LengthTwoKeepsOnlyBosEos()
    {
        var batch = new TextBatchPreparer(Tokenizer(), 2).Prepare(["", "abcd"], false);
        Assert.Equal(2, batch.SequenceLength);
        Assert.Equal(new long[] {101,102,101,102}, batch.InputIds.ToArray());
        Assert.Equal(new long[] {1,1,1,1}, batch.AttentionMask.ToArray());
        Assert.False(batch.HasTokenTypeIds);
        Assert.Empty(batch.TokenTypeIds.ToArray());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Prepare_EmptyBatchAndEmptyTextAreDistinct(bool types)
    {
        var preparer = new TextBatchPreparer(Tokenizer(), 20);
        var empty = preparer.Prepare([], types);
        Assert.Equal(0, empty.BatchSize);
        Assert.Equal(0, empty.SequenceLength);
        Assert.Empty(empty.InputIds.ToArray());
        Assert.Empty(empty.AttentionMask.ToArray());
        Assert.Empty(empty.TokenTypeIds.ToArray());
        Assert.Equal(types, empty.HasTokenTypeIds);
        var text = preparer.Prepare([""], types);
        Assert.Equal(1, text.BatchSize);
        Assert.Equal(2, text.SequenceLength);
        Assert.Equal(new long[] {101,102}, text.InputIds.ToArray());
        Assert.Equal(new long[] {1,1}, text.AttentionMask.ToArray());
        Assert.Equal(types, text.HasTokenTypeIds);
    }

    [Fact]
    public void Prepare_RejectsInvalidArgumentsAndTokenizerResults()
    {
        Assert.Throws<ArgumentNullException>(() => new TextTokenizer(null!, 0, 1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TextTokenizer(_ => [], -1, 1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TextTokenizer(_ => [], 0, -1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TextTokenizer(_ => [], 0, 1, -1));
        Assert.Throws<ArgumentNullException>(() => new TextBatchPreparer(null!, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TextBatchPreparer(Tokenizer(), 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TextBatchPreparer(Tokenizer(), 2, 0));
        Assert.Throws<ArgumentNullException>(() => Tokenizer().EncodeContent(null!));
        Assert.Throws<InvalidOperationException>(() => Tokenizer(_ => null!).EncodeContent("a"));
        var preparer = new TextBatchPreparer(Tokenizer(), 4);
        Assert.Throws<ArgumentNullException>(() => preparer.Prepare(null!));
        Assert.Throws<ArgumentNullException>(() => preparer.Prepare([null!]));
        Assert.Throws<InvalidOperationException>(() => new TextBatchPreparer(Tokenizer(_ => [-1]), 4).Prepare(["a"]));
    }

    [Fact]
    public void Prepare_CancellationStopsBeforeAndBetweenRows()
    {
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        int calls = 0;
        var preparer = new TextBatchPreparer(Tokenizer(_ => { calls++; return [8]; }), 5);
        Assert.Throws<OperationCanceledException>(() => preparer.Prepare(["a"], cancellationToken: cancelled.Token));
        Assert.Equal(0, calls);
        using var between = new CancellationTokenSource();
        preparer = new TextBatchPreparer(Tokenizer(_ => { calls++; between.Cancel(); return [8]; }), 5);
        Assert.Throws<OperationCanceledException>(() => preparer.Prepare(["a", "b"], cancellationToken: between.Token));
        Assert.Equal(1, calls);
        calls = 0;
        using var last = new CancellationTokenSource();
        preparer = new TextBatchPreparer(Tokenizer(_ => { calls++; last.Cancel(); return [8]; }), 5);
        Assert.Throws<OperationCanceledException>(() => preparer.Prepare(["a"], cancellationToken: last.Token));
        Assert.Equal(1, calls);
    }

    [Theory]
    [InlineData("\U0001F600\U0001F680 \U0001F469\u200D\U0001F4BB", 1, 1)]
    [InlineData("\U00010400\U00010428 \U0001D7D8\U0001D7D9", 1, 1)]
    [InlineData("hello \U0001F600 world", 5, 1, 6)]
    public void Bert_EmojiAndAstralTextPreserveUnknownWordGroups(string text, params int[] expected)
        => Assert.Equal(expected, Bert().EncodeContent(text));

    [Theory]
    [InlineData("$+=>", 24, 25, 26, 8)]
    [InlineData("\u00A3 \u20AC", 27, 28)]
    [InlineData("hello\u20ACworld", 1)]
    [InlineData("<s>one</s><pad><unk>  <mask> next", 7,9,8,10,7,11,9,8,7,12,8,7,13,8,7,14,8,15)]
    [InlineData("before \t\n<mask> after", 16, 7, 14, 8, 17)]
    public void Bert_AsciiAndUnicodeSymbolsAreNotDropped(string text, params int[] expected)
        => Assert.Equal(expected, Bert().EncodeContent(text));

    [Fact]
    public void Bert_SpecialTokensAreExtractedBeforeOrdinaryLowercasing()
    {
        const string text = "[CLS]Hello[SEP][PAD][UNK][MASK]";
        var tokenizer = Bert();
        Assert.Equal(new[] {2,5,3,0,1,4}, tokenizer.EncodeContent(text));
        var batch = new TextBatchPreparer(tokenizer, 20).Prepare([text, ""]);
        Assert.Equal(2, batch.BatchSize);
        Assert.Equal(8, batch.SequenceLength);
        Assert.Equal(new long[] {2,2,5,3,0,1,4,3, 2,3,0,0,0,0,0,0}, batch.InputIds.ToArray());
        // Literal [PAD] is content with mask=1; only the empty row's padding is masked.
        Assert.Equal(new long[] {1,1,1,1,1,1,1,1, 1,1,0,0,0,0,0,0}, batch.AttentionMask.ToArray());
        Assert.Equal(new long[16], batch.TokenTypeIds.ToArray());
    }

    [Theory]
    [InlineData("hello\tworld")]
    [InlineData("hello\nworld")]
    [InlineData("hello\rworld")]
    [InlineData("hello\t\r\nworld")]
    public void Bert_TabsAndNewlinesSeparateWordsRatherThanJoining(string text)
    {
        var tokenizer = Bert();
        Assert.Equal(new[] {5,6}, tokenizer.EncodeContent(text));
        Assert.Equal(new[] {1}, tokenizer.EncodeContent("helloworld"));
    }

    [Theory]
    [InlineData("[cls]Hello[sep][pad][unk][mask]", 18,20,19,5,18,21,19,18,12,19,18,13,19,18,14,19)]
    [InlineData("[Cls]Hello[MASK]", 18,20,19,5,4)]
    public void Bert_LowercaseSpecialLookingTextRemainsOrdinary(string text, params int[] expected)
        => Assert.Equal(expected, Bert().EncodeContent(text));

    [Theory]
    [InlineData("\uAC01", 29, 30, 31)]
    [InlineData("\u1100\u1161\u11A8", 29, 30, 31)]
    [InlineData("\uAC00", 29, 30)]
    public void Bert_HangulRemainsDecomposedForWordPiece(string text, params int[] expected)
    {
        // The fixture also contains composed syllables at IDs 32/33: NFC recomposition
        // would select those instead of the expected decomposed WordPiece sequence.
        Assert.Equal(expected, Bert().EncodeContent(text));
    }

    [Theory]
    [InlineData(0xD800)]
    [InlineData(0xDBFF)]
    [InlineData(0xDC00)]
    [InlineData(0xDFFF)]
    public void EncodeContent_InvalidUtf16IsRejectedBeforeDelegateOrPreparation(int codeUnit)
    {
        int calls = 0;
        var custom = Tokenizer(_ => { calls++; return [7]; });
        var bert = Bert();
        string invalid = new((char)codeUnit, 1);
        // Construct invalid code units at runtime: attribute-string serialization
        // can replace lone surrogates and accidentally test U+FFFD instead.
        foreach (string text in new[] {invalid, invalid + "hello", "hello" + invalid, "hello" + invalid + "world"})
        {
            var error = Assert.Throws<ArgumentException>(() => custom.EncodeContent(text));
            Assert.Equal("text", error.ParamName);
            Assert.Contains("UTF-16", error.Message);
            Assert.Throws<ArgumentException>(() => bert.EncodeContent(text));
            Assert.Throws<ArgumentException>(() => new TextBatchPreparer(custom, 8).Prepare([text]));
        }
        Assert.Equal(0, calls);
    }

    [Fact]
    public void EncodeContent_ValidSurrogatePairsReachDelegateUnchanged()
    {
        int calls = 0;
        string? received = null;
        var tokenizer = Tokenizer(text => { calls++; received = text; return [7,8]; });
        Assert.Equal(new[] {7,8}, tokenizer.EncodeContent("a\U0001F600b"));
        Assert.Equal("a\U0001F600b", received);
        Assert.Equal(1, calls);
    }

    private static TextTokenizer Bert()
        => TextTokenizer.CreateUncasedBert(Path.Combine(AppContext.BaseDirectory, "Fixtures", "bert-regression-vocab.txt"));
}
