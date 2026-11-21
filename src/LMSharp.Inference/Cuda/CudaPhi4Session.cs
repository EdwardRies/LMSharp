using static LMSharp.Inference.Cuda.CudaNative;

namespace LMSharp.Inference.Cuda;

internal sealed class CudaPhi4Session : IPhi4Session
{
    private const int BlockThreads = 128;
    private const int WarpsPerBlock = 4;
    private const int PrefillSyncInterval = 32;

    private readonly CudaPhi4Model model;
    private readonly Phi4Model.ModelShape shape;
    private readonly CudaDevice.DeviceAllocation workspace;
    private readonly ulong hidden;
    private readonly ulong normalized;
    private readonly ulong qkv;
    private readonly ulong attended;
    private readonly ulong projected;
    private readonly ulong gateUp;
    private readonly ulong activated;
    private readonly ulong rotation;
    private readonly ulong logits;
    private readonly ulong keys;
    private readonly ulong values;
    private readonly ulong layerCacheBytes;
    private readonly float[] output;

    public CudaPhi4Session(CudaPhi4Model model, int contextLength)
    {
        this.model = model;
        shape = model.Shape;
        Capacity = contextLength;
        output = new float[model.Tokenizer.VocabularySize];
        var cacheElements = (long)contextLength * shape.KvSize * shape.LayerCount;
        layerCacheBytes = (ulong)contextLength * (ulong)shape.KvSize * sizeof(float);
        var scratchElements = 4L * shape.EmbeddingSize + shape.QkvSize + 3L * shape.FeedForwardSize +
            shape.RotarySize + output.Length;
        workspace = model.Device.Allocate((scratchElements + 2 * cacheElements) * sizeof(float));
        var nextAddress = workspace.Address;
        hidden = Reserve(shape.EmbeddingSize);
        normalized = Reserve(shape.EmbeddingSize);
        qkv = Reserve(shape.QkvSize);
        attended = Reserve(shape.EmbeddingSize);
        projected = Reserve(shape.EmbeddingSize);
        gateUp = Reserve(2 * shape.FeedForwardSize);
        activated = Reserve(shape.FeedForwardSize);
        rotation = Reserve(shape.RotarySize);
        logits = Reserve(output.Length);
        keys = Reserve(cacheElements);
        values = Reserve(cacheElements);

        ulong Reserve(long elements)
        {
            var address = nextAddress;
            nextAddress += (ulong)elements * sizeof(float);
            return address;
        }
    }

    public int Capacity { get; }
    public int Position { get; private set; }

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Count == 0 || tokens.Count > Capacity - Position)
        {
            throw new ArgumentException("The prompt must be nonempty and fit in the remaining context.", nameof(tokens));
        }

        foreach (var token in tokens)
        {
            ValidateToken(token);
        }

        var start = Position;
        using var scope = model.Device.Bind();
        try
        {
            for (var index = 0; index < tokens.Count; index++)
            {
                Forward(tokens[index], cancellationToken);
                Position++;
                if ((index + 1) % PrefillSyncInterval == 0)
                {
                    Check(cuCtxSynchronize());
                }
            }

            ProjectLogits();
            return output;
        }
        catch
        {
            Position = start;
            throw;
        }
    }

    public ReadOnlySpan<float> Advance(int token, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        if (Position == Capacity)
        {
            throw new InvalidOperationException("The inference context is full.");
        }

        ValidateToken(token);
        using var scope = model.Device.Bind();
        Forward(token, cancellationToken);
        ProjectLogits();
        Position++;
        return output;
    }

    public void Rewind(int position)
    {
        ThrowIfDisposed();
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(position, Position);
        Position = position;
    }

    public void Dispose() => workspace.Dispose();

    private void Forward(int token, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var k = model.Kernels;
        var embeddings = model.Embeddings;
        Launch(k.Embedding, Blocks(shape.EmbeddingSize), BlockThreads,
            [embeddings.Address, hidden, (ulong)embeddings.Columns, (ulong)embeddings.RowBytes, embeddings.Type, (ulong)token]);
        var half = (ulong)(shape.RotarySize / 2);
        Launch(k.PrepareRope, 1, BlockThreads,
            [model.Frequencies, rotation, half, (ulong)Position, Bits(shape.RopeScale)]);
        var length = Position + 1;
        for (var index = 0; index < model.Layers.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var layer = model.Layers[index];
            var layerKeys = keys + (ulong)index * layerCacheBytes;
            var layerValues = values + (ulong)index * layerCacheBytes;

            Normalize(layer.AttentionNorm);
            Multiply(layer.Qkv, normalized, qkv);
            Launch(k.RopeAndCache, Blocks(shape.EmbeddingSize), BlockThreads,
                [qkv, layerKeys, layerValues, rotation, (ulong)shape.EmbeddingSize, (ulong)shape.KvSize,
                 (ulong)shape.HeadSize, half, (ulong)Position]);
            Launch(k.Attention, shape.Heads, BlockThreads,
                [qkv, layerKeys, layerValues, attended, (ulong)shape.HeadSize, (ulong)shape.KvSize,
                 (ulong)(shape.Heads / shape.KvHeads), (ulong)length, Bits(1 / MathF.Sqrt(shape.HeadSize))],
                sharedBytes: length * sizeof(float));
            Multiply(layer.AttentionOutput, attended, projected);
            AddResidual();

            Normalize(layer.FeedForwardNorm);
            Multiply(layer.GateUp, normalized, gateUp);
            Launch(k.SwiGlu, Blocks(shape.FeedForwardSize), BlockThreads,
                [gateUp, activated, (ulong)shape.FeedForwardSize]);
            Multiply(layer.Down, activated, projected);
            AddResidual();
        }
    }

    private void ProjectLogits()
    {
        Normalize(model.OutputNorm);
        Multiply(model.Embeddings, normalized, logits);
        Check(cuCtxSynchronize());
        CudaDevice.Download<float>(output, logits);
    }

    private void Normalize(ulong weights) => Launch(model.Kernels.RmsNorm, 1, BlockThreads,
        [hidden, weights, normalized, (ulong)shape.EmbeddingSize, Bits(shape.RmsEpsilon)]);

    private void Multiply(CudaPhi4Model.MatrixBuffer matrix, ulong source, ulong destination) =>
        Launch(model.Kernels.MatVec, (matrix.Rows + WarpsPerBlock - 1) / WarpsPerBlock, BlockThreads,
            [matrix.Address, source, destination, (ulong)matrix.Columns, (ulong)matrix.Rows, (ulong)matrix.RowBytes, matrix.Type]);

    private void AddResidual() => Launch(model.Kernels.Residual, Blocks(shape.EmbeddingSize), BlockThreads,
        [hidden, projected, (ulong)shape.EmbeddingSize]);

    private void ValidateToken(int token)
    {
        if ((uint)token >= (uint)output.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(token));
        }
    }

    private void ThrowIfDisposed()
    {
        model.ThrowIfDisposed();
        ObjectDisposedException.ThrowIf(workspace.Address == 0, this);
    }

    private static int Blocks(int elements) => (elements + BlockThreads - 1) / BlockThreads;
    private static ulong Bits(float value) => BitConverter.SingleToUInt32Bits(value);
}
