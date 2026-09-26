using CommunityToolkit.Embeddings.Onnx;
using Microsoft.ML.Tokenizers;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

public sealed class TokenizerExtensionsTests
{
    internal static Tokenizer Tokenizer(Func<string, IReadOnlyList<int>>? encode = null)
        => new CallbackTokenizer(encode ?? (text => text.Select(c => (int)c).ToArray()));

    private static TokenSequenceOptions Options(int length, int pad = 9) => new(length, 101, 102, pad);

    private sealed class CallbackTokenizer(Func<string, IReadOnlyList<int>> encode) : Tokenizer
    {
        protected override EncodeResults<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
            => new() { Tokens = encode(text ?? textSpan.ToString()), CharsConsumed = text?.Length ?? textSpan.Length };

        protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
            => throw new NotSupportedException("This preparation collaborator implements only content IDs.");

        public override System.Buffers.OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten)
            => throw new NotSupportedException("Preparation never decodes content.");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(9)]
    public void Prepare_BudgetsSpecialsTruncatesAndPadsInOrder(int pad)
    {
        var tokenizer = Tokenizer();
        var batch = tokenizer.PrepareBatch(["a", "bc", "xyz", "ghij"], Options(5, pad), 4);
        Assert.Equal(4, batch.BatchSize);
        Assert.Equal(5, batch.SequenceLength);
        Assert.Equal(new long[] {101,97,102,pad,pad, 101,98,99,102,pad, 101,120,121,122,102, 101,103,104,105,102}, batch.InputIds.ToArray());
        Assert.Equal(new long[] {1,1,1,0,0, 1,1,1,1,0, 1,1,1,1,1, 1,1,1,1,1}, batch.AttentionMask.ToArray());
        Assert.True(batch.HasTokenTypeIds);
        Assert.Equal(new long[20], batch.TokenTypeIds.ToArray());
        Assert.Throws<ArgumentException>(() => tokenizer.PrepareBatch(["a","b","c","d","e"], Options(5, pad), 4));
    }

    [Fact]
    public void Prepare_LengthTwoKeepsOnlyBosEos()
    {
        var batch = Tokenizer().PrepareBatch(["", "abcd"], Options(2), includeTokenTypeIds: false);
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
        var tokenizer = Tokenizer();
        var empty = tokenizer.PrepareBatch([], Options(20), includeTokenTypeIds: types);
        Assert.Equal(0, empty.BatchSize);
        Assert.Equal(0, empty.SequenceLength);
        Assert.Empty(empty.InputIds.ToArray());
        Assert.Empty(empty.AttentionMask.ToArray());
        Assert.Empty(empty.TokenTypeIds.ToArray());
        Assert.Equal(types, empty.HasTokenTypeIds);
        var text = tokenizer.PrepareBatch([""], Options(20), includeTokenTypeIds: types);
        Assert.Equal(1, text.BatchSize);
        Assert.Equal(2, text.SequenceLength);
        Assert.Equal(new long[] {101,102}, text.InputIds.ToArray());
        Assert.Equal(new long[] {1,1}, text.AttentionMask.ToArray());
        Assert.Equal(types, text.HasTokenTypeIds);
    }

    [Fact]
    public void Prepare_RejectsInvalidArgumentsAndTokenizerResults()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TokenSequenceOptions(2, -1, 1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TokenSequenceOptions(2, 0, -1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TokenSequenceOptions(2, 0, 1, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TokenSequenceOptions(1, 0, 1, 2));
        Assert.Throws<ArgumentNullException>(() => TokenizerExtensions.PrepareBatch(null!, [], Options(2)));
        Assert.Throws<ArgumentNullException>(() => Tokenizer().PrepareBatch([], null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => Tokenizer().PrepareBatch([], Options(2), 0));
        Assert.Throws<InvalidOperationException>(() => Tokenizer(_ => null!).PrepareBatch(["a"], Options(4)));
        var tokenizer = Tokenizer();
        Assert.Throws<ArgumentNullException>(() => tokenizer.PrepareBatch(null!, Options(4)));
        Assert.Throws<ArgumentNullException>(() => tokenizer.PrepareBatch([null!], Options(4)));
        Assert.Throws<InvalidOperationException>(() => Tokenizer(_ => [-1]).PrepareBatch(["a"], Options(4)));
    }

    [Fact]
    public void Prepare_CancellationStopsBeforeAndBetweenRows()
    {
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        int calls = 0;
        var tokenizer = Tokenizer(_ => { calls++; return [8]; });
        Assert.Throws<OperationCanceledException>(() => tokenizer.PrepareBatch(["a"], Options(5), cancellationToken: cancelled.Token));
        Assert.Equal(0, calls);
        using var between = new CancellationTokenSource();
        tokenizer = Tokenizer(_ => { calls++; between.Cancel(); return [8]; });
        Assert.Throws<OperationCanceledException>(() => tokenizer.PrepareBatch(["a", "b"], Options(5), cancellationToken: between.Token));
        Assert.Equal(1, calls);
        calls = 0;
        using var last = new CancellationTokenSource();
        tokenizer = Tokenizer(_ => { calls++; last.Cancel(); return [8]; });
        Assert.Throws<OperationCanceledException>(() => tokenizer.PrepareBatch(["a"], Options(5), cancellationToken: last.Token));
        Assert.Equal(1, calls);
    }

    [Theory]
    [InlineData("\U0001F600\U0001F680 \U0001F469\u200D\U0001F4BB", 1, 1)]
    [InlineData("\U00010400\U00010428 \U0001D7D8\U0001D7D9", 1, 1)]
    [InlineData("hello \U0001F600 world", 5, 1, 6)]
    public void Bert_EmojiAndAstralTextPreserveUnknownWordGroups(string text, params int[] expected)
        => Assert.Equal(expected, Bert().EncodeToIds(text));

    [Theory]
    [InlineData("$+=>", 24, 25, 26, 8)]
    [InlineData("\u00A3 \u20AC", 27, 28)]
    [InlineData("hello\u20ACworld", 1)]
    [InlineData("<s>one</s><pad><unk>  <mask> next", 7,9,8,10,7,11,9,8,7,12,8,7,13,8,7,14,8,15)]
    [InlineData("before \t\n<mask> after", 16, 7, 14, 8, 17)]
    public void Bert_AsciiAndUnicodeSymbolsAreNotDropped(string text, params int[] expected)
        => Assert.Equal(expected, Bert().EncodeToIds(text));

    [Fact]
    public void Bert_SpecialTokensAreExtractedBeforeOrdinaryLowercasing()
    {
        const string text = "[CLS]Hello[SEP][PAD][UNK][MASK]";
        var tokenizer = Bert();
        Assert.Equal(new[] {2,5,3,0,1,4}, tokenizer.EncodeToIds(text));
        Tokenizer standard = tokenizer;
        Assert.Equal(new[] {2,5,3,0,1,4}, standard.EncodeToIds(text));
        Assert.Equal(new[] {2,5,3,0,1,4}, tokenizer.EncodeToIds(text.AsSpan()));
        Assert.Equal(new[] {2,5,3,0,1,4}, standard.EncodeToIds(text.AsSpan()));
        var batch = tokenizer.PrepareBatch([text, ""], new(20, 2, 3, 0));
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
        Assert.Equal(new[] {5,6}, tokenizer.EncodeToIds(text));
        Assert.Equal(new[] {1}, tokenizer.EncodeToIds("helloworld"));
    }

    [Theory]
    [InlineData("[cls]Hello[sep][pad][unk][mask]", 18,20,19,5,18,21,19,18,12,19,18,13,19,18,14,19)]
    [InlineData("[Cls]Hello[MASK]", 18,20,19,5,4)]
    public void Bert_LowercaseSpecialLookingTextRemainsOrdinary(string text, params int[] expected)
        => Assert.Equal(expected, Bert().EncodeToIds(text));

    [Theory]
    [InlineData("\uAC01", 29, 30, 31)]
    [InlineData("\u1100\u1161\u11A8", 29, 30, 31)]
    [InlineData("\uAC00", 29, 30)]
    public void Bert_HangulRemainsDecomposedForWordPiece(string text, params int[] expected)
    {
        // The fixture also contains composed syllables at IDs 32/33: NFC recomposition
        // would select those instead of the expected decomposed WordPiece sequence.
        Assert.Equal(expected, Bert().EncodeToIds(text));
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
            var error = Assert.Throws<ArgumentException>(() => custom.PrepareBatch([text], Options(8)));
            Assert.Equal("text", error.ParamName);
            Assert.Contains("UTF-16", error.Message);
            Assert.Throws<ArgumentException>(() => bert.EncodeToIds(text));
            Assert.Throws<ArgumentException>(() => bert.EncodeToIds(text.AsSpan()));
        }
        Assert.Equal(0, calls);
    }

    [Fact]
    public void EncodeContent_ValidSurrogatePairsReachDelegateUnchanged()
    {
        int calls = 0;
        string? received = null;
        var tokenizer = Tokenizer(text => { calls++; received = text; return [7,8]; });
        var batch = tokenizer.PrepareBatch(["a\U0001F600b"], Options(8));
        Assert.Equal(new long[] {101,7,8,102}, batch.InputIds.ToArray());
        Assert.Equal(new long[] {1,1,1,1}, batch.AttentionMask.ToArray());
        Assert.Equal("a\U0001F600b", received);
        Assert.Equal(1, calls);
    }

    [Fact]
    public void Prepare_AcceptsOrdinaryWordPieceTokenizerAndExposesSequenceOptions()
    {
        Tokenizer tokenizer = WordPieceTokenizer.Create(BertVocabulary, new WordPieceOptions { UnknownToken = "[UNK]" });
        var options = new TokenSequenceOptions(5, 2, 3, 1);
        Assert.Equal(5, options.MaximumSequenceLength);
        var batch = tokenizer.PrepareBatch(["hello", ""], options);
        Assert.Equal(new long[] {2,5,3, 2,3,1}, batch.InputIds.ToArray());
        Assert.Equal(new long[] {1,1,1, 1,1,0}, batch.AttentionMask.ToArray());
        Assert.Equal(new long[6], batch.TokenTypeIds.ToArray());
    }

    [Fact]
    public void PrepareBatch_DefaultLimitIs32AndOldPublicPreparerIsRemoved()
    {
        Tokenizer tokenizer = Tokenizer(_ => [7]);
        var accepted = tokenizer.PrepareBatch(Enumerable.Repeat("x", 32).ToArray(), Options(3, 1));
        Assert.Equal(32, accepted.BatchSize);
        Assert.Equal(3, accepted.SequenceLength);
        Assert.Equal(Enumerable.Repeat(new long[] {101,7,102}, 32).SelectMany(row => row), accepted.InputIds.ToArray());
        Assert.Equal(Enumerable.Repeat(1L, 96), accepted.AttentionMask.ToArray());
        Assert.Throws<ArgumentException>(() => tokenizer.PrepareBatch(Enumerable.Repeat("x", 33).ToArray(), Options(3, 1)));
        Assert.Null(typeof(TokenizerExtensions).Assembly.GetType("CommunityToolkit.Embeddings.Onnx.TextBatchPreparer"));
        Assert.True(typeof(TokenizerExtensions).IsAbstract && typeof(TokenizerExtensions).IsSealed);
    }

    [Theory]
    [InlineData(int.MinValue)]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    public void SequenceOptions_RejectsLengthsBelowTwo(int length)
        => Assert.Throws<ArgumentOutOfRangeException>(() => new TokenSequenceOptions(length, 0, 0, 0));

    [Fact]
    public void SequenceOptions_ValidBoundariesAreImmutable()
    {
        var options = new TokenSequenceOptions(2, 0, int.MaxValue, 0);
        Assert.Equal(2, options.MaximumSequenceLength);
        Assert.Equal(0, options.BeginningTokenId);
        Assert.Equal(int.MaxValue, options.EndTokenId);
        Assert.Equal(0, options.PaddingTokenId);
        Assert.All(typeof(TokenSequenceOptions).GetProperties(), property => Assert.Null(property.SetMethod));
    }

    internal static string BertVocabulary => Path.Combine(AppContext.BaseDirectory, "Fixtures", "bert-regression-vocab.txt");
    private static BertUncasedTokenizer Bert() => new(BertVocabulary);
}
