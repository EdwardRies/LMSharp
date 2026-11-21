using System.Buffers.Binary;
using System.Collections.ObjectModel;
using System.Text;

namespace LMSharp.Gguf;

public sealed class GgufReader
{
    public const uint SupportedVersion = 3;

    private const ulong MaximumMetadataCount = 1_000_000;
    private const ulong MaximumArrayLength = 1_000_000;
    private const ulong MaximumStringBytes = 16 * 1024 * 1024;
    private const ulong MaximumTensorCount = 1_000_000;
    private const ulong MaximumTensorNameBytes = 64;
    private static readonly byte[] Magic = "GGUF"u8.ToArray();

    public async Task<GgufFile> ReadAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        await using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 64 * 1024,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        return await ReadAsync(stream, cancellationToken);
    }

    public async Task<GgufTensorFile> OpenTensorFileAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 64 * 1024, FileOptions.Asynchronous | FileOptions.RandomAccess);
        try
        {
            var info = await ReadAsync(stream, cancellationToken);
            return new GgufTensorFile(stream, info, cancellationToken);
        }
        catch
        {
            await stream.DisposeAsync();
            throw;
        }
    }

    public async Task<GgufFile> ReadAsync(Stream stream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead || !stream.CanSeek)
        {
            throw new ArgumentException("GGUF input must be a readable, seekable stream.", nameof(stream));
        }

        var reader = new PrimitiveReader(stream);
        var magic = await reader.ReadBytesAsync(Magic.Length, cancellationToken);
        if (!magic.AsSpan().SequenceEqual(Magic))
        {
            throw new GgufFormatException("Invalid GGUF magic bytes.");
        }

        var version = await reader.ReadUInt32Async(cancellationToken);
        if (version != SupportedVersion)
        {
            throw new GgufFormatException($"Unsupported GGUF version {version}; expected {SupportedVersion}.");
        }

        var tensorCount = await reader.ReadUInt64Async(cancellationToken);
        var metadataCount = await reader.ReadUInt64Async(cancellationToken);
        if (metadataCount > MaximumMetadataCount || metadataCount > reader.RemainingBytes / 12)
        {
            throw new GgufFormatException($"GGUF metadata count {metadataCount} is too large or truncated.");
        }

        var metadata = new Dictionary<string, GgufMetadataValue>(
            checked((int)metadataCount),
            StringComparer.Ordinal);
        for (ulong index = 0; index < metadataCount; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var key = await reader.ReadStringAsync(cancellationToken);
            if (string.IsNullOrEmpty(key))
            {
                throw new GgufFormatException("GGUF metadata contains an empty key.");
            }

            var type = ReadType(await reader.ReadUInt32Async(cancellationToken));
            GgufMetadataValue value;
            if (type == GgufValueType.Array)
            {
                var elementType = ReadType(await reader.ReadUInt32Async(cancellationToken));
                if (elementType == GgufValueType.Array)
                {
                    throw new GgufFormatException($"Nested GGUF arrays are not supported for '{key}'.");
                }

                var count = await reader.ReadUInt64Async(cancellationToken);
                if (count > MaximumArrayLength || count > reader.RemainingBytes / MinimumSize(elementType))
                {
                    throw new GgufFormatException($"GGUF array '{key}' is too large or truncated.");
                }

                var values = new object[checked((int)count)];
                for (var element = 0; element < values.Length; element++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    values[element] = await reader.ReadScalarAsync(elementType, cancellationToken);
                }

                value = new GgufMetadataValue(type, values, elementType);
            }
            else
            {
                value = new GgufMetadataValue(
                    type,
                    await reader.ReadScalarAsync(type, cancellationToken));
            }

            if (!metadata.TryAdd(key, value))
            {
                throw new GgufFormatException($"Duplicate GGUF metadata key '{key}'.");
            }
        }

        if (tensorCount > MaximumTensorCount || tensorCount > reader.RemainingBytes / 24)
        {
            throw new GgufFormatException($"GGUF tensor count {tensorCount} is too large or the file is truncated.");
        }

        var tensors = new List<GgufTensorDescriptor>(checked((int)tensorCount));
        var seenNames = new HashSet<string>(StringComparer.Ordinal);
        for (ulong index = 0; index < tensorCount; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var name = await reader.ReadStringAsync(cancellationToken, MaximumTensorNameBytes);
            if (string.IsNullOrEmpty(name))
            {
                throw new GgufFormatException("GGUF tensor descriptor contains an empty name.");
            }

            if (!seenNames.Add(name))
            {
                throw new GgufFormatException($"GGUF tensor descriptor contains duplicate name '{name}'.");
            }

            var nDimensions = await reader.ReadUInt32Async(cancellationToken);
            if (nDimensions == 0 || nDimensions > 4)
            {
                throw new GgufFormatException($"GGUF tensor '{name}' has invalid dimension count {nDimensions}.");
            }

            var dimensions = new ulong[checked((int)nDimensions)];
            for (var i = 0; i < nDimensions; i++)
            {
                dimensions[i] = await reader.ReadUInt64Async(cancellationToken);
                if (dimensions[i] == 0)
                {
                    throw new GgufFormatException($"GGUF tensor '{name}' has zero-sized dimension {i}.");
                }
            }

            var ggmlType = await reader.ReadUInt32Async(cancellationToken);
            var offset = await reader.ReadUInt64Async(cancellationToken);

            tensors.Add(new GgufTensorDescriptor(name, Array.AsReadOnly(dimensions), ggmlType, offset));
        }

        return new GgufFile(
            version,
            tensorCount,
            new ReadOnlyDictionary<string, GgufMetadataValue>(metadata))
        {
            Tensors = new ReadOnlyCollection<GgufTensorDescriptor>(tensors),
        };
    }

    private static GgufValueType ReadType(uint rawType)
    {
        if (rawType > (uint)GgufValueType.Float64)
        {
            throw new GgufFormatException($"Unsupported GGUF metadata type {rawType}.");
        }

        return (GgufValueType)rawType;
    }

    private static ulong MinimumSize(GgufValueType type) => type switch
    {
        GgufValueType.UInt8 or GgufValueType.Int8 or GgufValueType.Bool => 1,
        GgufValueType.UInt16 or GgufValueType.Int16 => 2,
        GgufValueType.UInt32 or GgufValueType.Int32 or GgufValueType.Float32 => 4,
        GgufValueType.UInt64 or GgufValueType.Int64 or GgufValueType.Float64 or GgufValueType.String => 8,
        _ => throw new GgufFormatException($"Unsupported GGUF array element type {type}."),
    };

    private sealed class PrimitiveReader(Stream stream)
    {
        private readonly byte[] scalar = new byte[8];
        private static readonly UTF8Encoding StrictUtf8 = new(false, true);

        public ulong RemainingBytes => checked((ulong)(stream.Length - stream.Position));

        public async ValueTask<byte[]> ReadBytesAsync(int length, CancellationToken cancellationToken)
        {
            if ((ulong)length > RemainingBytes)
            {
                throw new GgufFormatException("GGUF file is truncated.");
            }

            var bytes = new byte[length];
            await ReadExactlyAsync(bytes, cancellationToken);
            return bytes;
        }

        public async ValueTask<uint> ReadUInt32Async(CancellationToken cancellationToken)
        {
            await ReadScalarBytesAsync(4, cancellationToken);
            return BinaryPrimitives.ReadUInt32LittleEndian(scalar);
        }

        public async ValueTask<ulong> ReadUInt64Async(CancellationToken cancellationToken)
        {
            await ReadScalarBytesAsync(8, cancellationToken);
            return BinaryPrimitives.ReadUInt64LittleEndian(scalar);
        }

        public async ValueTask<string> ReadStringAsync(
            CancellationToken cancellationToken,
            ulong maximumBytes = MaximumStringBytes)
        {
            var length = await ReadUInt64Async(cancellationToken);
            if (length > maximumBytes || length > RemainingBytes)
            {
                throw new GgufFormatException("GGUF string is too large or truncated.");
            }

            var bytes = await ReadBytesAsync(checked((int)length), cancellationToken);
            try
            {
                return StrictUtf8.GetString(bytes);
            }
            catch (DecoderFallbackException exception)
            {
                throw new GgufFormatException($"GGUF string contains invalid UTF-8: {exception.Message}");
            }
        }

        public async ValueTask<object> ReadScalarAsync(
            GgufValueType type,
            CancellationToken cancellationToken) => type switch
            {
                GgufValueType.UInt8 => await ReadUInt8Async(cancellationToken),
                GgufValueType.Int8 => unchecked((sbyte)await ReadUInt8Async(cancellationToken)),
                GgufValueType.UInt16 => await ReadUInt16Async(cancellationToken),
                GgufValueType.Int16 => unchecked((short)await ReadUInt16Async(cancellationToken)),
                GgufValueType.UInt32 => await ReadUInt32Async(cancellationToken),
                GgufValueType.Int32 => unchecked((int)await ReadUInt32Async(cancellationToken)),
                GgufValueType.Float32 => BitConverter.Int32BitsToSingle(
                    unchecked((int)await ReadUInt32Async(cancellationToken))),
                GgufValueType.Bool => await ReadBoolAsync(cancellationToken),
                GgufValueType.String => await ReadStringAsync(cancellationToken),
                GgufValueType.UInt64 => await ReadUInt64Async(cancellationToken),
                GgufValueType.Int64 => unchecked((long)await ReadUInt64Async(cancellationToken)),
                GgufValueType.Float64 => BitConverter.Int64BitsToDouble(
                    unchecked((long)await ReadUInt64Async(cancellationToken))),
                _ => throw new GgufFormatException($"Unsupported GGUF scalar type {type}."),
            };

        private async ValueTask<byte> ReadUInt8Async(CancellationToken cancellationToken)
        {
            await ReadScalarBytesAsync(1, cancellationToken);
            return scalar[0];
        }

        private async ValueTask<ushort> ReadUInt16Async(CancellationToken cancellationToken)
        {
            await ReadScalarBytesAsync(2, cancellationToken);
            return BinaryPrimitives.ReadUInt16LittleEndian(scalar);
        }

        private async ValueTask<bool> ReadBoolAsync(CancellationToken cancellationToken)
        {
            var value = await ReadUInt8Async(cancellationToken);
            return value switch
            {
                0 => false,
                1 => true,
                _ => throw new GgufFormatException($"Invalid GGUF boolean value {value}."),
            };
        }

        private async ValueTask ReadScalarBytesAsync(int length, CancellationToken cancellationToken)
        {
            if ((ulong)length > RemainingBytes)
            {
                throw new GgufFormatException("GGUF file is truncated.");
            }

            await ReadExactlyAsync(scalar.AsMemory(0, length), cancellationToken);
        }

        private async ValueTask ReadExactlyAsync(
            Memory<byte> buffer,
            CancellationToken cancellationToken)
        {
            try
            {
                await stream.ReadExactlyAsync(buffer, cancellationToken);
            }
            catch (EndOfStreamException)
            {
                throw new GgufFormatException("GGUF file is truncated.");
            }
        }
    }
}
