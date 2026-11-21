using System.Collections.ObjectModel;
using Microsoft.Win32.SafeHandles;

namespace LMSharp.Gguf;

public sealed record GgufTensorRange(GgufTensorDescriptor Descriptor, long FileOffset, long ByteLength);

/// <summary>Owns the open GGUF file handle and provides reads within validated tensor ranges.</summary>
public sealed class GgufTensorFile : IDisposable
{
    private readonly FileStream stream;
    private readonly SafeFileHandle handle;

    internal GgufTensorFile(FileStream stream, GgufFile info, CancellationToken cancellationToken)
    {
        this.stream = stream;
        Info = info;

        if (info.Tensors.Count == 0)
        {
            throw new GgufFormatException("The GGUF file contains no tensor weights.");
        }

        var alignment = ReadAlignment(info);
        var directoryEnd = (ulong)stream.Position;
        var dataOffset = directoryEnd + (alignment - directoryEnd % alignment) % alignment;
        var fileLength = (ulong)stream.Length;
        if (dataOffset > fileLength)
        {
            throw new GgufFormatException("The GGUF file ends before the aligned tensor data section.");
        }

        DataOffset = (long)dataOffset;
        var dataLength = fileLength - dataOffset;
        var ranges = new Dictionary<string, GgufTensorRange>(info.Tensors.Count, StringComparer.Ordinal);
        foreach (var tensor in info.Tensors)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (tensor.Offset % alignment != 0)
            {
                throw new GgufFormatException($"GGUF tensor '{tensor.Name}' has an unaligned offset.");
            }

            var byteLength = ComputeByteLength(tensor);
            if (tensor.Offset > dataLength || byteLength > dataLength - tensor.Offset)
            {
                throw new GgufFormatException($"GGUF tensor '{tensor.Name}' extends beyond the end of the file.");
            }

            ranges.Add(tensor.Name, new GgufTensorRange(tensor, (long)(dataOffset + tensor.Offset), (long)byteLength));
        }

        long previousEnd = DataOffset;
        foreach (var range in ranges.Values.OrderBy(range => range.FileOffset))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (range.FileOffset < previousEnd)
            {
                throw new GgufFormatException($"GGUF tensor '{range.Descriptor.Name}' overlaps another tensor.");
            }

            previousEnd = range.FileOffset + range.ByteLength;
        }

        Tensors = new ReadOnlyDictionary<string, GgufTensorRange>(ranges);
        handle = stream.SafeFileHandle;
    }

    public GgufFile Info { get; }
    public long DataOffset { get; }
    public IReadOnlyDictionary<string, GgufTensorRange> Tensors { get; }

    /// <summary>Fills the destination with raw bytes starting at an offset within the named tensor.</summary>
    public async ValueTask ReadTensorAsync(
        string name,
        long tensorByteOffset,
        Memory<byte> destination,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(handle.IsClosed, this);
        cancellationToken.ThrowIfCancellationRequested();
        var tensor = Tensors[name];
        if (tensorByteOffset < 0 || tensorByteOffset > tensor.ByteLength)
        {
            throw new ArgumentOutOfRangeException(nameof(tensorByteOffset));
        }

        if (destination.Length > tensor.ByteLength - tensorByteOffset)
        {
            throw new ArgumentOutOfRangeException(nameof(destination), "The read would extend past the tensor's end.");
        }

        var fileOffset = tensor.FileOffset + tensorByteOffset;
        while (!destination.IsEmpty)
        {
            var bytesRead = await RandomAccess.ReadAsync(handle, destination, fileOffset, cancellationToken);
            if (bytesRead == 0)
            {
                throw new GgufFormatException($"GGUF tensor '{name}' was truncated while reading its data.");
            }

            destination = destination[bytesRead..];
            fileOffset += bytesRead;
        }
    }

    public void Dispose() => stream.Dispose();

    private static uint ReadAlignment(GgufFile info)
    {
        if (!info.Metadata.TryGetValue("general.alignment", out var alignment))
        {
            return 32;
        }

        if (alignment.Type != GgufValueType.UInt32 || alignment.Value is not uint value || value == 0 || value % 8 != 0)
        {
            throw new GgufFormatException("GGUF general.alignment must be a positive UInt32 multiple of 8.");
        }

        return value;
    }

    private static ulong ComputeByteLength(GgufTensorDescriptor tensor)
    {
        if (!GgmlBlockLayout.TryGet((GgmlStorageType)tensor.GgmlType, out var layout))
        {
            throw new GgufFormatException($"GGUF tensor '{tensor.Name}' has unsupported GGML type {tensor.GgmlType}.");
        }

        var valuesPerBlock = (ulong)layout.ElementCount;
        var rowLength = tensor.Dimensions[0];
        if (rowLength % valuesPerBlock != 0)
        {
            throw new GgufFormatException(
                $"GGUF tensor '{tensor.Name}' has a row dimension that is not divisible by {valuesPerBlock}.");
        }

        try
        {
            var byteLength = checked(rowLength / valuesPerBlock * (ulong)layout.ByteCount);
            for (var dimension = 1; dimension < tensor.DimensionCount; dimension++)
            {
                byteLength = checked(byteLength * tensor.Dimensions[dimension]);
            }

            return byteLength;
        }
        catch (OverflowException)
        {
            throw new GgufFormatException($"GGUF tensor '{tensor.Name}' has a byte length that overflows UInt64.");
        }
    }
}
