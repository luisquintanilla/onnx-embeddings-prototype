using Microsoft.ML.Tokenizers;

namespace CommunityToolkit.Embeddings.Onnx;

/// <summary>Single-sequence tokenization, BOS/EOS budgeting, truncation and longest-in-batch right padding.</summary>
public sealed class TextBatchPreparer
{
    public TextBatchPreparer(Tokenizer tokenizer, TokenSequenceOptions sequenceOptions, int maximumBatchSize = 32)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(sequenceOptions);
        ArgumentOutOfRangeException.ThrowIfLessThan(maximumBatchSize, 1);
        Tokenizer = tokenizer;
        SequenceOptions = sequenceOptions;
        MaximumBatchSize = maximumBatchSize;
    }

    /// <summary>The standard tokenizer must emit content only, without automatically inserted surrounding tokens.</summary>
    public Tokenizer Tokenizer { get; }
    public TokenSequenceOptions SequenceOptions { get; }
    public int MaximumSequenceLength => SequenceOptions.MaximumSequenceLength;
    public int MaximumBatchSize { get; }

    public TokenBatch Prepare(IReadOnlyList<string> texts, bool includeTokenTypeIds = true, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        cancellationToken.ThrowIfCancellationRequested();
        if (texts.Count > MaximumBatchSize)
            throw new ArgumentException($"At most {MaximumBatchSize} texts may be prepared in one batch.", nameof(texts));
        if (texts.Count == 0) return new TokenBatch(0, 0, Array.Empty<long>(), Array.Empty<long>(), includeTokenTypeIds ? Array.Empty<long>() : null);

        var sequences = new IReadOnlyList<int>[texts.Count];
        int length = 2;
        for (int i = 0; i < texts.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ArgumentNullException.ThrowIfNull(texts[i]);
            TokenizerContract.ValidateText(texts[i]);
            sequences[i] = Tokenizer.EncodeToIds(texts[i])
                ?? throw new InvalidOperationException("The tokenizer returned null.");
            length = Math.Max(length, Math.Min(sequences[i].Count, MaximumSequenceLength - 2) + 2);
        }

        int count = checked(texts.Count * length);
        var ids = new long[count];
        var masks = new long[count];
        long[]? types = includeTokenTypeIds ? new long[count] : null;
        if (SequenceOptions.PaddingTokenId != 0) Array.Fill(ids, (long)SequenceOptions.PaddingTokenId);
        for (int row = 0; row < texts.Count; row++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int offset = row * length;
            int contentLength = Math.Min(sequences[row].Count, MaximumSequenceLength - 2);
            ids[offset] = SequenceOptions.BeginningTokenId;
            for (int i = 0; i < contentLength; i++)
            {
                int id = sequences[row][i];
                if (id < 0) throw new InvalidOperationException("The tokenizer returned a negative token ID.");
                ids[offset + i + 1] = id;
            }
            ids[offset + contentLength + 1] = SequenceOptions.EndTokenId;
            masks.AsSpan(offset, contentLength + 2).Fill(1);
        }
        return new TokenBatch(texts.Count, length, ids, masks, types);
    }
}
