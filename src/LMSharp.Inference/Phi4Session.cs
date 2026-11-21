using LMSharp.Tensors;

namespace LMSharp.Inference;

/// <summary>One sequential decoding session with its own working buffers and key/value cache.</summary>
public sealed class Phi4Session : IPhi4Session
{
    private readonly Phi4Model model;
    private readonly Phi4Model.ModelShape shape;
    private readonly LayerCache[] cache;
    private readonly Phi4RotaryEmbedding rope;
    private readonly ParallelOptions options;
    private readonly float attentionScale;
    private readonly float[] hidden;
    private readonly float[] normalized;
    private readonly float[] qkv;
    private readonly float[] attended;
    private readonly float[] projected;
    private readonly float[] scores;
    private readonly float[] gateUp;
    private readonly float[] activated;
    private readonly float[] logits;

    internal Phi4Session(Phi4Model model, int contextLength, int threads)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextLength);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(contextLength, model.MaximumContextLength);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(threads);
        this.model = model;
        shape = model.Shape;
        Capacity = contextLength;
        cache = new LayerCache[shape.LayerCount];
        for (var layer = 0; layer < cache.Length; layer++)
        {
            cache[layer] = new LayerCache(contextLength * shape.KvSize);
        }

        rope = new Phi4RotaryEmbedding(shape.FrequencyBase, shape.RopeScale, model.RopeFactors);
        options = new ParallelOptions { MaxDegreeOfParallelism = threads };
        attentionScale = 1 / MathF.Sqrt(shape.HeadSize);
        hidden = new float[shape.EmbeddingSize];
        normalized = new float[shape.EmbeddingSize];
        qkv = new float[shape.QkvSize];
        attended = new float[shape.EmbeddingSize];
        projected = new float[shape.EmbeddingSize];
        scores = new float[contextLength];
        gateUp = new float[2 * shape.FeedForwardSize];
        activated = new float[shape.FeedForwardSize];
        logits = new float[model.Tokenizer.VocabularySize];
    }

    public int Capacity { get; }
    public int Position { get; private set; }

    /// <summary>Appends a nonempty prompt and returns logits valid until the next session operation.</summary>
    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, CancellationToken cancellationToken = default) =>
        Prefill(tokens, cancellationToken, observeLayer: null);

    internal ReadOnlySpan<float> Prefill(
        IReadOnlyList<int> tokens, CancellationToken cancellationToken,
        Action<int, int, ReadOnlyMemory<float>>? observeLayer)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Count == 0 || tokens.Count > Capacity - Position)
        {
            throw new ArgumentException("The prompt must be nonempty and fit in the remaining context.", nameof(tokens));
        }

        foreach (var token in tokens)
        {
            ValidateToken(token);
        }

        options.CancellationToken = cancellationToken;
        var start = Position;
        try
        {
            for (var index = 0; index < tokens.Count; index++)
            {
                Forward(tokens[index], projectLogits: index == tokens.Count - 1, observeLayer);
            }

            return logits;
        }
        catch
        {
            // Cache entries at and after the restored position are overwritten when reused.
            Position = start;
            throw;
        }
    }

    /// <summary>Appends one token and returns logits valid until the next session operation.</summary>
    public ReadOnlySpan<float> Advance(int token, CancellationToken cancellationToken = default)
    {
        if (Position == Capacity)
        {
            throw new InvalidOperationException("The inference context is full.");
        }

        ValidateToken(token);
        options.CancellationToken = cancellationToken;
        Forward(token, projectLogits: true, observeLayer: null);
        return logits;
    }

    public void Reset() => Position = 0;

    internal void Rewind(int position)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(position, Position);
        Position = position;
    }

    void IPhi4Session.Rewind(int position) => Rewind(position);

    public void Dispose() { }

    private void Forward(int token, bool projectLogits, Action<int, int, ReadOnlyMemory<float>>? observeLayer)
    {
        options.CancellationToken.ThrowIfCancellationRequested();
        model.Embeddings.ReadRow(token, hidden);
        rope.SetPosition(Position);
        for (var index = 0; index < model.Layers.Length; index++)
        {
            var layer = model.Layers[index];
            CpuKernels.RmsNorm(hidden, layer.AttentionNorm, shape.RmsEpsilon, normalized);
            layer.Qkv.Multiply(normalized, qkv, options);
            Attend(cache[index]);
            layer.AttentionOutput.Multiply(attended, projected, options);
            AddResidual();

            CpuKernels.RmsNorm(hidden, layer.FeedForwardNorm, shape.RmsEpsilon, normalized);
            layer.GateUp.Multiply(normalized, gateUp, options);
            CpuKernels.SwiGlu(gateUp.AsSpan(0, shape.FeedForwardSize),
                gateUp.AsSpan(shape.FeedForwardSize), activated);
            layer.Down.Multiply(activated, projected, options);
            AddResidual();
            observeLayer?.Invoke(Position, index, hidden);
        }

        if (projectLogits)
        {
            CpuKernels.RmsNorm(hidden, model.OutputNorm, shape.RmsEpsilon, normalized);
            model.Embeddings.Multiply(normalized, logits, options);
        }

        Position++;
    }

    private void Attend(LayerCache layer)
    {
        var queries = qkv.AsSpan(0, shape.EmbeddingSize);
        var key = qkv.AsSpan(shape.EmbeddingSize, shape.KvSize);
        var value = qkv.AsSpan(shape.EmbeddingSize + shape.KvSize, shape.KvSize);
        rope.RotateHeads(queries, shape.HeadSize);
        rope.RotateHeads(key, shape.HeadSize);
        key.CopyTo(layer.Keys.AsSpan(Position * shape.KvSize));
        value.CopyTo(layer.Values.AsSpan(Position * shape.KvSize));

        var cacheScores = scores.AsSpan(0, Position + 1);
        var headsPerGroup = shape.Heads / shape.KvHeads;
        for (var head = 0; head < shape.Heads; head++)
        {
            var query = queries.Slice(head * shape.HeadSize, shape.HeadSize);
            var output = attended.AsSpan(head * shape.HeadSize, shape.HeadSize);
            var kvOffset = (head / headsPerGroup) * shape.HeadSize;
            for (var past = 0; past <= Position; past++)
            {
                var pastKey = layer.Keys.AsSpan(past * shape.KvSize + kvOffset, shape.HeadSize);
                cacheScores[past] = CpuKernels.Dot(query, pastKey) * attentionScale;
            }

            CpuKernels.SoftmaxInPlace(cacheScores);
            output.Clear();
            for (var past = 0; past <= Position; past++)
            {
                var pastValue = layer.Values.AsSpan(past * shape.KvSize + kvOffset, shape.HeadSize);
                var weight = cacheScores[past];
                for (var element = 0; element < output.Length; element++)
                {
                    output[element] += weight * pastValue[element];
                }
            }
        }
    }

    private void AddResidual()
    {
        for (var i = 0; i < hidden.Length; i++)
        {
            hidden[i] += projected[i];
        }
    }

    private void ValidateToken(int token)
    {
        if ((uint)token >= (uint)model.Tokenizer.VocabularySize)
        {
            throw new ArgumentOutOfRangeException(nameof(token));
        }
    }

    private sealed class LayerCache(int size)
    {
        public float[] Keys { get; } = GC.AllocateUninitializedArray<float>(size);
        public float[] Values { get; } = GC.AllocateUninitializedArray<float>(size);
    }
}
