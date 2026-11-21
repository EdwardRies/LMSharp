using System.Runtime.InteropServices;
using LMSharp.Gguf;

namespace LMSharp.Tensors;

/// <summary>A row-major matrix backed by the original GGML bytes.</summary>
public sealed class PackedMatrix
{
    private readonly ReadOnlyMemory<byte> weights;
    private readonly GgmlStorageType type;
    private readonly GgmlBlockLayout layout;
    private readonly int rowBytes;

    /// <summary>The caller must keep the supplied weights unchanged for the lifetime of the matrix.</summary>
    public PackedMatrix(GgmlStorageType type, int rows, int columns, ReadOnlyMemory<byte> weights)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        layout = GgmlBlockLayout.Get(type);
        if (columns % layout.ElementCount != 0)
        {
            throw new ArgumentException("Each row must contain full GGML blocks.", nameof(columns));
        }

        rowBytes = checked(columns / layout.ElementCount * layout.ByteCount);
        if (weights.Length != (long)rows * rowBytes)
        {
            throw new ArgumentException("Weight bytes do not match the matrix shape.", nameof(weights));
        }

        this.type = type;
        this.weights = weights;
        Rows = rows;
        Columns = columns;
    }

    public int Rows { get; }
    public int Columns { get; }

    public static async Task<PackedMatrix> LoadAsync(
        GgufTensorFile file, string name, CancellationToken cancellationToken = default)
    {
        var tensor = file.Tensors[name];
        var dimensions = tensor.Descriptor.Dimensions;
        if (dimensions.Count != 2 || dimensions.Any(dimension => dimension > int.MaxValue) ||
            tensor.ByteLength > Array.MaxLength)
        {
            throw new GgufFormatException($"GGUF tensor '{name}' cannot be loaded as a managed matrix.");
        }

        var weights = GC.AllocateUninitializedArray<byte>((int)tensor.ByteLength);
        await file.ReadTensorAsync(name, 0, weights, cancellationToken);
        return new PackedMatrix((GgmlStorageType)tensor.Descriptor.GgmlType,
            rows: (int)dimensions[1], columns: (int)dimensions[0], weights);
    }

    public void ReadRow(int row, Span<float> output)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, Rows);
        if (output.Length != Columns)
        {
            throw new ArgumentException("Destination length must match the column count.", nameof(output));
        }

        GgmlDequantizer.Decode(type, GetRowBytes(row), output);
    }

    public void Multiply(ReadOnlyMemory<float> input, Memory<float> output, ParallelOptions options)
    {
        if (input.Length != Columns || output.Length != Rows || input.Span.Overlaps(output.Span))
        {
            throw new ArgumentException("Matrix-vector operands must match the matrix shape and have separate buffers.");
        }

        if (options.MaxDegreeOfParallelism == 1 || Rows < 64)
        {
            for (var row = 0; row < Rows; row++)
            {
                options.CancellationToken.ThrowIfCancellationRequested();
                output.Span[row] = DotRow(row, input.Span);
            }

            return;
        }

        Parallel.For(0, Rows, options, row => output.Span[row] = DotRow(row, input.Span));
    }

    private ReadOnlySpan<byte> GetRowBytes(int row) => weights.Span.Slice(row * rowBytes, rowBytes);

    private float DotRow(int row, ReadOnlySpan<float> input)
    {
        var bytes = GetRowBytes(row);
        if (type == GgmlStorageType.F32 && BitConverter.IsLittleEndian)
        {
            return CpuKernels.Dot(input, MemoryMarshal.Cast<byte, float>(bytes));
        }

        Span<float> block = stackalloc float[layout.ElementCount];
        var sum = 0f;
        while (!input.IsEmpty)
        {
            GgmlDequantizer.DecodeBlock(type, bytes[..layout.ByteCount], block);
            sum += CpuKernels.Dot(input[..layout.ElementCount], block);
            input = input[layout.ElementCount..];
            bytes = bytes[layout.ByteCount..];
        }

        return sum;
    }
}
