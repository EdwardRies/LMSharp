using LMSharp.Gguf;
using LMSharp.Tokenization;
using static LMSharp.Inference.Cuda.CudaNative;

namespace LMSharp.Inference.Cuda;

internal sealed class CudaPhi4Model : IPhi4Model
{
    private readonly nint module;
    private readonly CudaDevice.DeviceAllocation weights;
    private bool disposed;

    private CudaPhi4Model(CudaDevice device, nint module, CudaDevice.DeviceAllocation weights,
        Phi4WeightLayout layout, long dataOffset, long frequencyOffset)
    {
        Device = device;
        this.module = module;
        this.weights = weights;
        Shape = layout.Shape;
        Tokenizer = layout.Tokenizer;
        Frequencies = weights.Address + (ulong)frequencyOffset;
        OutputNorm = Address(layout.OutputNorm);
        Embeddings = Matrix(layout.Embeddings);
        Layers = layout.Layers.Select(layer => new Layer(
            Address(layer.AttentionNorm), Matrix(layer.Qkv), Matrix(layer.AttentionOutput),
            Address(layer.FeedForwardNorm), Matrix(layer.GateUp), Matrix(layer.Down))).ToArray();

        using var scope = device.Bind();
        Kernels = new KernelSet(
            Function("embedding"), Function("matvec"), Function("rms_norm"), Function("prepare_rope"),
            Function("rope_and_cache"), Function("attention"), Function("residual"), Function("swiglu"));

        ulong Address(GgufTensorRange range) => weights.Address + (ulong)(range.FileOffset - dataOffset);

        MatrixBuffer Matrix(GgufTensorRange range)
        {
            var columns = (int)range.Descriptor.Dimensions[0];
            var rows = (int)range.Descriptor.Dimensions[1];
            return new MatrixBuffer(Address(range), columns, rows, (int)(range.ByteLength / rows),
                range.Descriptor.GgmlType);
        }

        nint Function(string name)
        {
            Check(cuModuleGetFunction(out var function, module, name));
            return function;
        }
    }

    public Phi4Tokenizer Tokenizer { get; }
    public int MaximumContextLength => Shape.ShortContextLength;
    public string DeviceName => $"CUDA (VRAM) · {Device.Name}";
    internal CudaDevice Device { get; }
    internal Phi4Model.ModelShape Shape { get; }
    internal MatrixBuffer Embeddings { get; }
    internal ulong OutputNorm { get; }
    internal ulong Frequencies { get; }
    internal Layer[] Layers { get; }
    internal KernelSet Kernels { get; }

    public static async Task<CudaPhi4Model> LoadAsync(string path, CancellationToken cancellationToken)
    {
        var device = new CudaDevice();
        nint module = 0;
        CudaDevice.DeviceAllocation? weights = null;
        try
        {
            using var file = await new GgufReader().OpenTensorFileAsync(path, cancellationToken);
            var layout = Phi4WeightLayout.Read(file);
            var factors = await layout.ReadRopeFactorsAsync(file, cancellationToken);
            var tensors = layout.ModelTensors().OrderBy(tensor => tensor.FileOffset).ToArray();
            var dataLength = tensors.Max(tensor => tensor.FileOffset + tensor.ByteLength) - file.DataOffset;
            var frequencyOffset = checked((dataLength + 255) & ~255L);
            module = device.LoadModule(CudaCompiler.Compile(device.ArchitectureTarget));
            weights = device.Allocate(checked(frequencyOffset + factors.Length * sizeof(float)));

            var staging = new byte[16 * 1024 * 1024];
            foreach (var tensor in tensors)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var destination = weights.Address + (ulong)(tensor.FileOffset - file.DataOffset);
                if (tensor.Descriptor.DimensionCount == 1)
                {
                    var values = await Phi4WeightLayout.ReadVectorAsync(file, tensor, cancellationToken);
                    device.Upload<float>(destination, values);
                    continue;
                }

                for (long offset = 0; offset < tensor.ByteLength; offset += staging.Length)
                {
                    var chunk = staging.AsMemory(0, (int)Math.Min(staging.Length, tensor.ByteLength - offset));
                    await file.ReadTensorAsync(tensor.Descriptor.Name, offset, chunk, cancellationToken);
                    device.Upload<byte>(destination + (ulong)offset, chunk.Span);
                }
            }

            for (var index = 0; index < factors.Length; index++)
            {
                factors[index] = MathF.Pow(layout.Shape.FrequencyBase, -(float)index / factors.Length) / factors[index];
            }

            device.Upload<float>(weights.Address + (ulong)frequencyOffset, factors);
            return new CudaPhi4Model(device, module, weights, layout, file.DataOffset, frequencyOffset);
        }
        catch
        {
            Release(device, module, weights);
            throw;
        }
    }

    public IPhi4Session CreateSession(int contextLength, int? threads = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextLength);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(contextLength, MaximumContextLength);
        ThrowIfDisposed();
        return new CudaPhi4Session(this, contextLength);
    }

    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);

    public void Dispose()
    {
        if (!disposed)
        {
            disposed = true;
            Release(Device, module, weights);
        }
    }

    private static void Release(CudaDevice device, nint module, CudaDevice.DeviceAllocation? weights)
    {
        try
        {
            weights?.Dispose();
        }
        finally
        {
            try
            {
                if (module != 0)
                {
                    using var scope = device.Bind();
                    Check(cuModuleUnload(module));
                }
            }
            finally
            {
                device.Dispose();
            }
        }
    }

    internal readonly record struct MatrixBuffer(ulong Address, int Columns, int Rows, int RowBytes, uint Type);
    internal sealed record Layer(
        ulong AttentionNorm, MatrixBuffer Qkv, MatrixBuffer AttentionOutput,
        ulong FeedForwardNorm, MatrixBuffer GateUp, MatrixBuffer Down);
    internal sealed record KernelSet(
        nint Embedding, nint MatVec, nint RmsNorm, nint PrepareRope, nint RopeAndCache,
        nint Attention, nint Residual, nint SwiGlu);
}
