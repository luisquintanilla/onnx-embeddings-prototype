using Microsoft.ML.Tokenizers;

namespace CommunityToolkit.Embeddings.Onnx;

/// <summary>Single-sequence tokenization, BOS/EOS budgeting, truncation and longest-in-batch right padding.</summary>
public static class TokenizerExtensions
{
    /// <summary>Prepares owned inputs from content-only tokenization. The tokenizer must not insert surrounding tokens.</summary>
    public static TokenBatch PrepareBatch(this Tokenizer tokenizer, IReadOnlyList<string> texts,
        TokenSequenceOptions sequenceOptions, int maximumBatchSize = 32, bool includeTokenTypeIds = true,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(sequenceOptions);
        ArgumentOutOfRangeException.ThrowIfLessThan(maximumBatchSize, 1);
        ArgumentNullException.ThrowIfNull(texts);
        cancellationToken.ThrowIfCancellationRequested();
        if (texts.Count > maximumBatchSize)
            throw new ArgumentException($"At most {maximumBatchSize} texts may be prepared in one batch.", nameof(texts));
        if (texts.Count == 0) return new TokenBatch(0, 0, Array.Empty<long>(), Array.Empty<long>(), includeTokenTypeIds ? Array.Empty<long>() : null);

        int maximumSequenceLength = sequenceOptions.MaximumSequenceLength;
        var sequences = new IReadOnlyList<int>[texts.Count];
        int length = 2;
        for (int i = 0; i < texts.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ArgumentNullException.ThrowIfNull(texts[i]);
            TokenizerContract.ValidateText(texts[i]);
            sequences[i] = tokenizer.EncodeToIds(texts[i])
                ?? throw new InvalidOperationException("The tokenizer returned null.");
            length = Math.Max(length, Math.Min(sequences[i].Count, maximumSequenceLength - 2) + 2);
        }

        int count = checked(texts.Count * length);
        var ids = new long[count];
        var masks = new long[count];
        long[]? types = includeTokenTypeIds ? new long[count] : null;
        if (sequenceOptions.PaddingTokenId != 0) Array.Fill(ids, (long)sequenceOptions.PaddingTokenId);
        for (int row = 0; row < texts.Count; row++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int offset = row * length;
            int contentLength = Math.Min(sequences[row].Count, maximumSequenceLength - 2);
            ids[offset] = sequenceOptions.BeginningTokenId;
            for (int i = 0; i < contentLength; i++)
            {
                int id = sequences[row][i];
                if (id < 0) throw new InvalidOperationException("The tokenizer returned a negative token ID.");
                ids[offset + i + 1] = id;
            }
            ids[offset + contentLength + 1] = sequenceOptions.EndTokenId;
            masks.AsSpan(offset, contentLength + 2).Fill(1);
        }
        return new TokenBatch(texts.Count, length, ids, masks, types);
    }
}
