using System.Buffers;
using System.Reflection;
using System.Text.Json;
using CommunityToolkit.Embeddings.Onnx;
using Microsoft.ML.Tokenizers;

namespace CommunityToolkit.Embeddings.Onnx.Tests;

public sealed class TokenizerContractTests
{
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);
    private static BertUncasedTokenizer Bert() => new(Fixture("bert-contract-vocab.txt"));
    private static Granite30MEnglishTokenizer Granite()
        => new(Fixture("granite-contract-vocab.json"), Fixture("granite-contract-merges.txt"));
    private static Tokenizer Adapter(bool granite) => granite ? Granite() : Bert();

    [Fact]
    public void Adapters_AreSealedStandardTokenizersWithoutHiddenEncodingOverloads()
    {
        foreach (Type type in new[] {typeof(BertUncasedTokenizer), typeof(Granite30MEnglishTokenizer)})
        {
            Assert.Equal(typeof(Tokenizer), type.BaseType);
            Assert.True(type.IsSealed);
            Assert.DoesNotContain(type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly),
                method => method.Name.StartsWith("Encode", StringComparison.Ordinal) ||
                          method.Name.StartsWith("Count", StringComparison.Ordinal) ||
                          method.Name.StartsWith("GetIndex", StringComparison.Ordinal));
        }
        Assert.Null(typeof(BertUncasedTokenizer).Assembly.GetType("CommunityToolkit.Embeddings.Onnx.TextTokenizer"));
        Assert.Null(Granite().Normalizer);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Adapters_IdsTokenValuesCountAndDecodeMatchIndependentSyntheticHf(bool granite)
    {
        using var reference = JsonDocument.Parse(File.ReadAllText(Fixture("tokenizer-contract-goldens.json")));
        Assert.Equal("4.57.6", reference.RootElement.GetProperty("packages").GetProperty("transformers").GetString());
        Assert.Equal("0.22.2", reference.RootElement.GetProperty("packages").GetProperty("tokenizers").GetString());
        Tokenizer tokenizer = Adapter(granite);
        var rows = reference.RootElement.GetProperty(granite ? "granite" : "bert");
        Assert.Equal(8, rows.GetArrayLength());
        foreach (var row in rows.EnumerateArray())
        {
            string text = row.GetProperty("text").GetString()!;
            int[] ids = row.GetProperty("ids").EnumerateArray().Select(value => value.GetInt32()).ToArray();
            string[] values = row.GetProperty("tokens").EnumerateArray().Select(value => value.GetString()!).ToArray();
            Assert.Equal(ids, tokenizer.EncodeToIds(text));
            Assert.Equal(ids, tokenizer.EncodeToIds(text.AsSpan()));
            Assert.Equal(ids, tokenizer.EncodeToTokens(text, out _).Select(token => token.Id));
            Assert.Equal(values, tokenizer.EncodeToTokens(text.AsSpan(), out _).Select(token => token.Value));
            Assert.Equal(ids.Length, tokenizer.CountTokens(text));
            Assert.Equal(ids.Length, tokenizer.CountTokens(text.AsSpan()));
            // HF cleanup is off. These particular BERT rows agree with the promised
            // Microsoft span-decoder wordpiece/space semantics, not general HF cleanup.
            AssertDecoded(tokenizer, ids, row.GetProperty("decoded").GetString()!);
        }
    }

    [Fact]
    public void Bert_NormalizedOffsetsAccountForAccentRemovalWhitespaceAndPreservedSpecials()
    {
        AssertTokens(Bert(), "Cafe\u0301\tHello!", "cafe hello!",
            [9,5,8], ["cafe","hello","!"], [(0,4),(5,10),(10,11)]);
        AssertTokens(Bert(), "Café中[CLS]Hello!", "cafe 中 [CLS]hello!",
            [9,14,2,5,8], ["cafe","中","[CLS]","hello","!"], [(0,4),(5,6),(7,12),(12,17),(17,18)]);
        AssertTokens(Bert(), "a\0b", "ab", [20,21], ["a","##b"], [(0,1),(1,2)]);
        AssertTokens(Bert(), "😀 hello", "😀 hello", [26,5], ["😀","hello"], [(0,2),(3,8)]);
    }

    [Fact]
    public void Bert_NormalizationFalsePreservesCaseAccentControlsAndOriginalOffsets()
    {
        AssertTokens(Bert(), "Café Hello!", null, [11,10,8], ["Café","Hello","!"],
            [(0,4),(5,10),(10,11)], normalization: false);
        AssertTokens(Bert(), "a\0b", null, [1], ["[UNK]"], [(0,3)], normalization: false);
        AssertTokens(Bert(), "Cafe\u0301", null, [1], ["[UNK]"], [(0,5)], normalization: false);
        AssertTokens(Bert(), "[CLS]Hello[SEP]", null, [2,10,3], ["[CLS]","Hello","[SEP]"],
            [(0,5),(5,10),(10,15)], normalization: false);
    }

    [Fact]
    public void Bert_PreTokenizationFalseDisablesAllSplittingButWholeSpecialRemainsKnown()
    {
        AssertTokens(Bert(), "hello!", "hello!", [5,8], ["hello","!"], [(0,5),(5,6)]);
        AssertTokens(Bert(), "hello!", "hello!", [23], ["hello!"], [(0,6)], preTokenization: false);
        AssertTokens(Bert(), "hello world", "hello world", [1], ["[UNK]"], [(0,11)], preTokenization: false);
        AssertTokens(Bert(), "[CLS]Hello![SEP]", "[CLS]hello![SEP]",
            [1], ["[UNK]"], [(0,16)], preTokenization: false);
        AssertTokens(Bert(), "[CLS]Hello[SEP]", null,
            [1], ["[UNK]"], [(0,15)],
            preTokenization: false, normalization: false);
        AssertTokens(Bert(), "[CLS]", "[CLS]", [2], ["[CLS]"], [(0,5)], preTokenization: false);
    }

    [Theory]
    [InlineData(1, 1, 6, 1, 10)]
    [InlineData(2, 2, 9, 2, 6)]
    [InlineData(3, 3, 17, 3, 0)]
    [InlineData(4, 3, 17, 3, 0)]
    public void Bert_BoundedWordPiecesAndReverseIndexFollowAdvertisedOffsets(
        int budget, int prefixCount, int forward, int suffixCount, int reverse)
    {
        // "##ing" begins inside the word: slicing/reencoding is not an ID promise.
        AssertBoundaries(Bert(), "  playing hello  ", "  playing hello  ",
            [12,13,5], budget, prefixCount, forward, suffixCount, reverse);
    }

    [Fact]
    public void Bert_BoundedNormalizationDisabledUsesOriginalCoordinates()
        => AssertBoundaries(Bert(), "Café Hello!  ", null, [11,10,8], 2, 2, 10, 2, 5, normalization: false);

    [Fact]
    public void Adapters_BoundedQueriesHonorDisabledPreTokenizationAndNormalization()
    {
        AssertBoundaries(Bert(), "[CLS]Hello![SEP]", "[CLS]hello![SEP]",
            [1], 1, 1, 16, 1, 0, preTokenization: false);
        AssertBoundaries(Bert(), "[CLS]Hello[SEP]", null,
            [1], 1, 1, 15, 1, 0, normalization: false, preTokenization: false);
        AssertBoundaries(Granite(), "<s>hello!</s>", null,
            [64,119,66,264,64,51,119,66], 4, 4, 9, 4, 9, normalization: false, preTokenization: false);
    }

    [Fact]
    public void Granite_OriginalUtf16OffsetsCoverWholeScalarsAndByteTokensMayOverlap()
    {
        AssertTokens(Granite(), "Aé😀Z", null,
            [69,199,173,244,163,156,132,94], ["A","Ã","©","ð","Ł","ĺ","Ģ","Z"],
            [(0,1),(1,2),(1,2),(2,4),(2,4),(2,4),(2,4),(4,5)]);
        AssertTokens(Granite(), "𐐀𝟘", null,
            [244,148,148,132,244,161,163,156], ["ð","Ĳ","Ĳ","Ģ","ð","Ŀ","Ł","ĺ"],
            [(0,2),(0,2),(0,2),(0,2),(2,4),(2,4),(2,4),(2,4)]);
        AssertTokens(Granite(), " \t\r\n", null, [36,13,17,14], ["Ġ","ĉ","č","Ċ"], [(0,1),(1,2),(2,3),(3,4)]);
    }

    [Theory]
    [InlineData(1, 1, 1, 1, 4)]
    [InlineData(2, 1, 1, 1, 4)]
    [InlineData(3, 3, 2, 1, 4)]
    [InlineData(4, 3, 2, 1, 4)]
    [InlineData(5, 3, 2, 5, 2)]
    [InlineData(6, 3, 2, 5, 2)]
    [InlineData(7, 7, 4, 7, 1)]
    [InlineData(8, 8, 5, 8, 0)]
    [InlineData(9, 8, 5, 8, 0)]
    public void Granite_BoundedEncodingAndIndicesNeverSplitOverlappingScalarTokens(
        int budget, int prefixCount, int forward, int suffixCount, int reverse)
        => AssertBoundaries(Granite(), "Aé😀Z", null,
            [69,199,173,244,163,156,132,94], budget, prefixCount, forward, suffixCount, reverse);

    [Fact]
    public void Granite_InsufficientScalarBudgetIsEmptyButPreparationUsesExactContentPrefix()
    {
        var tokenizer = Granite();
        AssertBoundaries(tokenizer, "😀", null, [244,163,156,132], 3, 0, 0, 0, 2);
        var batch = new TextBatchPreparer(tokenizer, new(5, 0, 2, 1)).Prepare(["😀", ""]);
        // Preparation intentionally takes THREE bytes of this four-token scalar.
        // Calling bounded EncodeToIds instead would incorrectly drop all its content.
        Assert.Equal(new long[] {0,244,163,156,2, 0,2,1,1,1}, batch.InputIds.ToArray());
        Assert.Equal(new long[] {1,1,1,1,1, 1,1,0,0,0}, batch.AttentionMask.ToArray());
        Assert.Equal(new long[10], batch.TokenTypeIds.ToArray());
    }

    [Fact]
    public void Granite_PreTokenizationFalseDisablesAllSplittingButWholeSpecialRemainsKnown()
    {
        AssertTokens(Granite(), "hello!", null, [263,37], ["hello","!"], [(0,5),(5,6)]);
        AssertTokens(Granite(), "hello!", null, [264], ["hello!"], [(0,6)], preTokenization: false);
        AssertTokens(Granite(), "<s>hello!</s>", null,
            [64,119,66,264,64,51,119,66], ["<","s",">","hello!","<","/","s",">"],
            [(0,1),(1,2),(2,3),(3,9),(9,10),(10,11),(11,12),(12,13)],
            preTokenization: false, normalization: false);
        AssertTokens(Granite(), "<mask>", null, [50264], ["<mask>"], [(0,6)], preTokenization: false);
        AssertTokens(Granite(), "A \t<mask>é", null,
            [69,50264,199,173], ["A","<mask>","Ã","©"], [(0,1),(3,9),(9,10),(9,10)]);
        // No normalizer exists: toggling normalization must not alter case or accents.
        AssertTokens(Granite(), "Aé", null, [69,199,173], ["A","Ã","©"],
            [(0,1),(1,2),(1,2)], normalization: false);
    }

    [Fact]
    public void Adapters_ExposeConfiguredNormalizerAndPreTokenizerObjectsWithConcreteBehavior()
    {
        var bert = Bert();
        var normalizer = Assert.IsAssignableFrom<Normalizer>(bert.Normalizer);
        Assert.Equal("[CLS]cafe hello[SEP]", normalizer.Normalize("[CLS]Café\tHello[SEP]"));
        Assert.Equal("[CLS]cafe hello[SEP]", normalizer.Normalize("[CLS]Café\tHello[SEP]".AsSpan()));
        var bertSplitter = Assert.IsAssignableFrom<PreTokenizer>(bert.PreTokenizer);
        (int, int)[] bertExpected = [(0,5),(5,1),(6,5),(11,5)];
        Assert.Equal(bertExpected, bertSplitter.PreTokenize("hello![CLS]world"));
        Assert.Equal(bertExpected, bertSplitter.PreTokenize("hello![CLS]world".AsSpan()));
        var granite = Granite();
        Assert.Null(granite.Normalizer);
        var byteSplitter = Assert.IsAssignableFrom<PreTokenizer>(granite.PreTokenizer);
        (int, int)[] byteExpected = [(0,5),(5,1),(6,3)];
        Assert.Equal(byteExpected, byteSplitter.PreTokenize("hello!<s>"));
        Assert.Equal(byteExpected, byteSplitter.PreTokenize("hello!<s>".AsSpan()));
    }

    [Fact]
    public void Granite_ConcreteAndBaseCallsLeaveSurroundingTokensToPreparation()
    {
        var concrete = Granite();
        Tokenizer standard = concrete;
        const string text = "<s>hello</s><pad><unk><mask>";
        int[] expected = [0,263,2,1,3,50264];
        Assert.Equal(expected, concrete.EncodeToIds(text));
        Assert.Equal(expected, standard.EncodeToIds(text));
        Assert.Equal(expected, concrete.EncodeToIds(text.AsSpan()));
        Assert.Equal(expected, standard.EncodeToIds(text.AsSpan()));
        var batch = new TextBatchPreparer(standard, new(20, 0, 2, 1)).Prepare([text, ""]);
        Assert.Equal(new long[] {0,0,263,2,1,3,50264,2, 0,2,1,1,1,1,1,1}, batch.InputIds.ToArray());
        Assert.Equal(new long[] {1,1,1,1,1,1,1,1, 1,1,0,0,0,0,0,0}, batch.AttentionMask.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Adapters_NullEmptyAndDefaultSpanHaveTheSameEmptyContentContract(bool granite)
    {
        Tokenizer tokenizer = Adapter(granite);
        Assert.Empty(tokenizer.EncodeToIds((string)null!));
        Assert.Empty(tokenizer.EncodeToIds(""));
        Assert.Empty(tokenizer.EncodeToIds(default(ReadOnlySpan<char>)));
        Assert.Empty(tokenizer.EncodeToTokens((string)null!, out _));
        Assert.Empty(tokenizer.EncodeToTokens(default(ReadOnlySpan<char>), out _));
        Assert.Equal(0, tokenizer.CountTokens((string)null!));
        Assert.Equal(0, tokenizer.CountTokens(default(ReadOnlySpan<char>)));
        // Upstream may bypass normalization for empty input; null and "" both
        // advertise the same empty coordinate space. No nonempty value is valid.
        Assert.Empty(tokenizer.EncodeToIds("", 1, out string? normalized, out int consumed));
        Assert.True(string.IsNullOrEmpty(normalized));
        Assert.Equal(0, consumed);
        Assert.Equal(0, tokenizer.GetIndexByTokenCount("", 1, out normalized, out consumed));
        Assert.True(string.IsNullOrEmpty(normalized));
        Assert.Equal(0, consumed);
        Assert.Equal(0, tokenizer.GetIndexByTokenCountFromEnd("", 1, out normalized, out consumed));
        Assert.True(string.IsNullOrEmpty(normalized));
        Assert.Equal(0, consumed);
        Assert.Empty(tokenizer.EncodeToIds(default(ReadOnlySpan<char>), 1, out normalized, out consumed));
        Assert.True(string.IsNullOrEmpty(normalized));
        Assert.Equal(0, consumed);
        AssertDecoded(tokenizer, [], "");
    }

    [Fact]
    public void Bert_WhitespaceOnlyFullConsumptionIncludesTrailingWhitespace()
        => AssertBoundaries(Bert(), " \t\r\n", "    ", [], 1, 0, 4, 0, 0);

    [Theory]
    [InlineData(false, 0)]
    [InlineData(false, -1)]
    [InlineData(false, int.MinValue)]
    [InlineData(true, 0)]
    [InlineData(true, -1)]
    [InlineData(true, int.MinValue)]
    public void Adapters_NonPositiveBudgetsThrowForEmptyAndNonemptyStringAndSpan(bool granite, int budget)
    {
        Tokenizer tokenizer = Adapter(granite);
        foreach (string text in new[] {"", "hello"})
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.EncodeToIds(text, budget, out _, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.EncodeToIds(text.AsSpan(), budget, out _, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.GetIndexByTokenCount(text, budget, out _, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.GetIndexByTokenCount(text.AsSpan(), budget, out _, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.GetIndexByTokenCountFromEnd(text, budget, out _, out _));
            Assert.Throws<ArgumentOutOfRangeException>(() => tokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), budget, out _, out _));
        }
    }

    [Theory]
    [InlineData(false, 0xD800)]
    [InlineData(false, 0xDC00)]
    [InlineData(true, 0xD800)]
    [InlineData(true, 0xDC00)]
    public void Adapters_MalformedUtf16RejectedAcrossEveryEncodingAndQuerySurface(bool granite, int codeUnit)
    {
        Tokenizer tokenizer = Adapter(granite);
        foreach (string text in new[] {new string((char)codeUnit, 1), "a" + (char)codeUnit, (char)codeUnit + "hello"})
        foreach (bool normalization in new[] {true, false})
        foreach (bool preTokenization in new[] {true, false})
        {
            Assert.Throws<ArgumentException>(() => tokenizer.EncodeToIds(text, preTokenization, normalization));
            Assert.Throws<ArgumentException>(() => tokenizer.EncodeToIds(text.AsSpan(), preTokenization, normalization));
            Assert.Throws<ArgumentException>(() => tokenizer.EncodeToTokens(text, out _, preTokenization, normalization));
            Assert.Throws<ArgumentException>(() => tokenizer.EncodeToTokens(text.AsSpan(), out _, preTokenization, normalization));
            Assert.Throws<ArgumentException>(() => tokenizer.CountTokens(text, preTokenization, normalization));
            Assert.Throws<ArgumentException>(() => tokenizer.CountTokens(text.AsSpan(), preTokenization, normalization));
            Assert.Throws<ArgumentException>(() => tokenizer.EncodeToIds(text, 1, out _, out _, preTokenization, normalization));
            Assert.Throws<ArgumentException>(() => tokenizer.EncodeToIds(text.AsSpan(), 1, out _, out _, preTokenization, normalization));
            Assert.Throws<ArgumentException>(() => tokenizer.GetIndexByTokenCount(text, 1, out _, out _, preTokenization, normalization));
            Assert.Throws<ArgumentException>(() => tokenizer.GetIndexByTokenCount(text.AsSpan(), 1, out _, out _, preTokenization, normalization));
            Assert.Throws<ArgumentException>(() => tokenizer.GetIndexByTokenCountFromEnd(text, 1, out _, out _, preTokenization, normalization));
            Assert.Throws<ArgumentException>(() => tokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), 1, out _, out _, preTokenization, normalization));
        }
    }

    [Fact]
    public void Bert_DecodeUsesWordPieceSpacingPreservesSpecialsAndCommitsWholeTokens()
    {
        var tokenizer = Bert();
        AssertDecoded(tokenizer, [2,5,7,6,8,3], "[CLS] hello , world ! [SEP]");
        AssertDecoded(tokenizer, [12,13], "playing");
        AssertPartial(tokenizer, [5,6], 4, OperationStatus.DestinationTooSmall, 0, "");
        AssertPartial(tokenizer, [5,6], 5, OperationStatus.DestinationTooSmall, 1, "hello");
        AssertPartial(tokenizer, [5,6], 6, OperationStatus.DestinationTooSmall, 1, "hello");
        AssertPartial(tokenizer, [5,6], 11, OperationStatus.Done, 2, "hello world");
        AssertPartial(tokenizer, [12,13], 6, OperationStatus.DestinationTooSmall, 1, "play");
        AssertPartial(tokenizer, [12,13], 7, OperationStatus.Done, 2, "playing");
        AssertPartial(tokenizer, [26], 1, OperationStatus.DestinationTooSmall, 0, "");
        AssertPartial(tokenizer, [26], 2, OperationStatus.Done, 1, "😀");
        AssertDecoded(tokenizer, [27], "supercalifragilisticexpialidocious");
    }

    [Theory]
    [InlineData(0, 0, "")]
    [InlineData(1, 1, "A")]
    [InlineData(2, 3, "Aé")]
    [InlineData(3, 3, "Aé")]
    [InlineData(4, 7, "Aé😀")]
    [InlineData(5, 8, "Aé😀Z")]
    public void Granite_DecodePartialDestinationsCommitOnlyWholeUtf8Groups(int capacity, int idsConsumed, string decoded)
        => AssertPartial(Granite(), [69,199,173,244,163,156,132,94], capacity,
            capacity == 5 ? OperationStatus.Done : OperationStatus.DestinationTooSmall, idsConsumed, decoded);

    [Fact]
    public void Granite_DecodeMergedTokenAndAddedTokenRemainAtomic()
    {
        AssertPartial(Granite(), [263,37], 4, OperationStatus.DestinationTooSmall, 0, "");
        AssertPartial(Granite(), [263,37], 5, OperationStatus.DestinationTooSmall, 1, "hello");
        AssertPartial(Granite(), [50264], 5, OperationStatus.DestinationTooSmall, 0, "");
        AssertPartial(Granite(), [50264], 6, OperationStatus.Done, 1, "<mask>");
        AssertDecoded(Granite(), [0,263,2,1,3,50264], "<s>hello</s><pad><unk><mask>");
    }

    [Fact]
    public void Granite_DecodeTokensContainingPartialAndCompleteScalarsCommitAsOneGroup()
    {
        // Tiny synthetic merges: 265="XÃ" -> bytes 58 C3; 266="©Y" -> A9 59.
        // The first token already contains X but cannot commit any of its bytes
        // until a later ID completes é; the second also contains a trailing Y.
        AssertDecoded(Granite(), [265,266], "XéY");
        AssertPartial(Granite(), [265,266], 2, OperationStatus.DestinationTooSmall, 0, "");
        AssertPartial(Granite(), [265,266], 3, OperationStatus.Done, 2, "XéY");
        AssertPartial(Granite(), [69,265,266], 3, OperationStatus.DestinationTooSmall, 1, "A");
        AssertPartial(Granite(), [69,265], 20, OperationStatus.InvalidData, 1, "A");
        Assert.Throws<InvalidOperationException>(() => Granite().Decode([265]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Adapters_StringDecodeEnumeratesInputOnlyOnceIncludingBufferGrowth(bool granite)
    {
        int enumerations = 0;
        IEnumerable<int> Once()
        {
            Assert.Equal(1, ++enumerations);
            yield return granite ? 50264 : 27;
        }
        // The BERT token is34 UTF16 chars: longer than the initial one-ID buffer,
        // so a decoder which re-enumerates instead of snapshotting fails this test.
        Assert.Equal(granite ? "<mask>" : "supercalifragilisticexpialidocious", Adapter(granite).Decode(Once()));
        Assert.Equal(1, enumerations);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Adapters_DecodeInvalidIdsReportFailureAndOnlyCommittedPrefix(bool granite)
    {
        Tokenizer tokenizer = Adapter(granite);
        int valid = granite ? 69 : 5;
        string prefix = granite ? "A" : "hello";
        foreach (int bad in new[] {-1, int.MaxValue, 50000})
        {
            Assert.Throws<InvalidOperationException>(() => tokenizer.Decode([bad]));
            Assert.Throws<InvalidOperationException>(() => tokenizer.Decode([valid,bad]));
            AssertPartial(tokenizer, [bad], 40, OperationStatus.InvalidData, 0, "");
            AssertPartial(tokenizer, [valid,bad], 40, OperationStatus.InvalidData, 1, prefix);
        }
        Assert.Throws<ArgumentNullException>(() => tokenizer.Decode(null!));
        Assert.Throws<ArgumentNullException>(() => tokenizer.Decode(null!, new char[10], out _, out _));
    }

    [Fact]
    public void Granite_StrictUtf8RejectsIncompleteOverlongSurrogateAndOrphanBytes()
    {
        var tokenizer = Granite();
        // Synthetic byte IDs are byte+4, so these cases do not depend on .NET encoding.
        foreach (int[] invalid in new int[][]
        {
            [199], [244,163,156], [173], [199,69], // incomplete, orphan, wrong continuation
            [196,132], [241,164,132], [248,148,132,132] // overlong, surrogate, >U+10FFFF
        })
        {
            Assert.Throws<InvalidOperationException>(() => tokenizer.Decode(invalid));
            AssertPartial(tokenizer, invalid, 20, OperationStatus.InvalidData, 0, "");
            AssertPartial(tokenizer, new[] {69}.Concat(invalid).ToArray(), 20, OperationStatus.InvalidData, 1, "A");
        }
        AssertDecoded(tokenizer, [199,173], "é");
        AssertDecoded(tokenizer, [244,163,156,132], "😀");
    }

    private static void AssertTokens(Tokenizer tokenizer, string text, string? normalized, int[] ids,
        string[] values, (int Start, int End)[] offsets, bool preTokenization = true, bool normalization = true)
    {
        Assert.Equal(ids, tokenizer.EncodeToIds(text, preTokenization, normalization));
        Assert.Equal(ids, tokenizer.EncodeToIds(text.AsSpan(), preTokenization, normalization));
        var fromString = tokenizer.EncodeToTokens(text, out string? stringNormalized, preTokenization, normalization);
        var fromSpan = tokenizer.EncodeToTokens(text.AsSpan(), out string? spanNormalized, preTokenization, normalization);
        Assert.Equal(normalized, stringNormalized);
        Assert.Equal(normalized, spanNormalized);
        foreach (var tokens in new[] {fromString, fromSpan})
        {
            Assert.Equal(ids, tokens.Select(token => token.Id));
            Assert.Equal(values, tokens.Select(token => token.Value));
            Assert.Equal(offsets, tokens.Select(token => (token.Offset.Start.Value, token.Offset.End.Value)));
            Assert.All(tokens, token => { Assert.False(token.Offset.Start.IsFromEnd); Assert.False(token.Offset.End.IsFromEnd); });
        }
        Assert.Equal(ids.Length, tokenizer.CountTokens(text, preTokenization, normalization));
        Assert.Equal(ids.Length, tokenizer.CountTokens(text.AsSpan(), preTokenization, normalization));
    }

    private static void AssertBoundaries(Tokenizer tokenizer, string text, string? normalized, int[] full,
        int budget, int prefixCount, int forward, int suffixCount, int reverse,
        bool normalization = true, bool preTokenization = true)
    {
        Assert.Equal(full, tokenizer.EncodeToIds(text, preTokenization, normalization));
        Assert.Equal(full.Take(prefixCount), tokenizer.EncodeToIds(text, budget, out string? n1, out int c1, preTokenization, normalization));
        Assert.Equal(full.Take(prefixCount), tokenizer.EncodeToIds(text.AsSpan(), budget, out string? n2, out int c2, preTokenization, normalization));
        Assert.Equal((normalized, forward), (n1, c1));
        Assert.Equal((normalized, forward), (n2, c2));
        Assert.Equal(forward, tokenizer.GetIndexByTokenCount(text, budget, out n1, out c1, preTokenization, normalization));
        Assert.Equal(forward, tokenizer.GetIndexByTokenCount(text.AsSpan(), budget, out n2, out c2, preTokenization, normalization));
        Assert.Equal((normalized, prefixCount), (n1, c1));
        Assert.Equal((normalized, prefixCount), (n2, c2));
        Assert.Equal(reverse, tokenizer.GetIndexByTokenCountFromEnd(text, budget, out n1, out c1, preTokenization, normalization));
        Assert.Equal(reverse, tokenizer.GetIndexByTokenCountFromEnd(text.AsSpan(), budget, out n2, out c2, preTokenization, normalization));
        Assert.Equal((normalized, suffixCount), (n1, c1));
        Assert.Equal((normalized, suffixCount), (n2, c2));
        Assert.Equal(full.Length, tokenizer.CountTokens(text, preTokenization, normalization));
        Assert.Equal(full.Length, tokenizer.CountTokens(text.AsSpan(), preTokenization, normalization));
    }

    private static void AssertDecoded(Tokenizer tokenizer, int[] ids, string expected)
    {
        Assert.Equal(expected, tokenizer.Decode(ids));
        AssertPartial(tokenizer, ids, expected.Length, OperationStatus.Done, ids.Length, expected);
    }

    private static void AssertPartial(Tokenizer tokenizer, int[] ids, int capacity,
        OperationStatus expectedStatus, int expectedConsumed, string expectedText)
    {
        char[] destination = Enumerable.Repeat('\u25A1', capacity + 2).ToArray();
        OperationStatus status = tokenizer.Decode(ids, destination.AsSpan(0, capacity), out int consumed, out int written);
        Assert.Equal(expectedStatus, status);
        Assert.Equal(expectedConsumed, consumed);
        Assert.Equal(expectedText.Length, written);
        Assert.Equal(expectedText, new string(destination, 0, written));
        // Not even a separator or half-surrogate may leak beyond a committed group.
        Assert.All(destination.Skip(written), value => Assert.Equal('\u25A1', value));
    }
}
