using System.Text;
using LMSharp.Gguf;
using Xunit;

namespace LMSharp.UnitTests;

public sealed class GgufReaderTests
{
    [Fact]
    public async Task ReadsValidHeaderScalarsAndArrayMetadata()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 0, metadataCount: 3);
            WriteString(writer, "general.architecture");
            writer.Write((uint)GgufValueType.String);
            WriteString(writer, "phi3");
            WriteString(writer, "phi3.block_count");
            writer.Write((uint)GgufValueType.UInt32);
            writer.Write(32u);
            WriteString(writer, "general.tags");
            writer.Write((uint)GgufValueType.Array);
            writer.Write((uint)GgufValueType.String);
            writer.Write(2UL);
            WriteString(writer, "nlp");
            WriteString(writer, "code");
        });

        var result = await new GgufReader().ReadAsync(stream);

        Assert.Equal(3u, result.Version);
        Assert.Equal(0UL, result.TensorCount);
        Assert.Equal("phi3", result.Architecture);
        Assert.Equal(32u, result.Metadata["phi3.block_count"].Value);
        Assert.Equal(GgufValueType.String, result.Metadata["general.tags"].ElementType);
        Assert.Equal(new object[] { "nlp", "code" }, result.Metadata["general.tags"].Value);
    }

    [Fact]
    public async Task ReadsMultipleTensorDescriptors()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 2, metadataCount: 0);
            WriteString(writer, "tensor1");
            writer.Write(2u);
            writer.Write(100UL);
            writer.Write(200UL);
            writer.Write(1u);
            writer.Write(0UL);
            WriteString(writer, "tensor2");
            writer.Write(3u);
            writer.Write(50UL);
            writer.Write(75UL);
            writer.Write(100UL);
            writer.Write(2u);
            writer.Write(400UL);
        });

        var result = await new GgufReader().ReadAsync(stream);

        Assert.Equal(2UL, result.TensorCount);
        Assert.Equal(2, result.Tensors.Count);
        Assert.Equal("tensor1", result.Tensors[0].Name);
        Assert.Equal(new ulong[] { 100UL, 200UL }, result.Tensors[0].Dimensions);
        Assert.Equal(1u, result.Tensors[0].GgmlType);
        Assert.Equal(0UL, result.Tensors[0].Offset);
        Assert.Equal("tensor2", result.Tensors[1].Name);
        Assert.Equal(new ulong[] { 50UL, 75UL, 100UL }, result.Tensors[1].Dimensions);
        Assert.Equal(2u, result.Tensors[1].GgmlType);
        Assert.Equal(400UL, result.Tensors[1].Offset);
    }

    [Fact]
    public async Task RejectsInvalidMagic()
    {
        using var stream = CreateFile(writer =>
        {
            writer.Write(Encoding.ASCII.GetBytes("BAD!"));
            writer.Write(3u);
            writer.Write(0UL);
            writer.Write(0UL);
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Contains("magic", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RejectsUnsupportedVersion()
    {
        using var stream = CreateFile(writer =>
        {
            writer.Write(Encoding.ASCII.GetBytes("GGUF"));
            writer.Write(2u);
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Contains("Unsupported GGUF version 2", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RejectsTruncatedHeader()
    {
        using var stream = CreateFile(writer => writer.Write(Encoding.ASCII.GetBytes("GGUF")));

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Contains("truncated", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RejectsTruncatedTensorDirectory()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.True(
            exception.Message.Contains("tensor", StringComparison.OrdinalIgnoreCase) ||
            exception.Message.Contains("truncat", StringComparison.OrdinalIgnoreCase),
            $"Expected message to mention 'tensor' or 'truncat', but was: {exception.Message}");
    }

    [Fact]
    public async Task RejectsTruncatedMetadataString()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 0, metadataCount: 1);
            writer.Write(20UL);
            writer.Write(Encoding.UTF8.GetBytes("short"));
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Contains("truncated", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RejectsTensorDescriptorWithEmptyName()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
            WriteString(writer, ""); // Empty name
            writer.Write(1u);
            writer.Write(100UL);
            writer.Write(1u);
            writer.Write(0UL);
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Contains("empty name", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RejectsTensorDescriptorNameLongerThan64Bytes()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
            WriteString(writer, new string('n', 65));
            writer.Write(1u);
            writer.Write(100UL);
            writer.Write(1u);
            writer.Write(0UL);
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Contains("too large", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RejectsTensorDescriptorWithInvalidDimensionCount()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
            WriteString(writer, "tensor");
            writer.Write(5u);
            writer.Write(0UL); // Clear the directory-size preflight.
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Contains("invalid dimension count", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RejectsTensorDescriptorWithTruncatedDimensions()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
            WriteString(writer, "tensor");
            writer.Write(3u);
            writer.Write(100UL);
            writer.Write(200UL);
            writer.Write(300u); // Only half of the third 64-bit dimension.
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Equal("GGUF file is truncated.", exception.Message);
    }

    [Fact]
    public async Task ReadsFileWithZeroTensors()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 0, metadataCount: 0);
        });

        var result = await new GgufReader().ReadAsync(stream);

        Assert.Equal(0UL, result.TensorCount);
        Assert.Empty(result.Tensors);
    }

    [Fact]
    public async Task RejectsTensorDescriptorWithDuplicateName()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 2, metadataCount: 0);
            WriteString(writer, "tensor");
            writer.Write(1u);
            writer.Write(100UL);
            writer.Write(1u);
            writer.Write(0UL);
            WriteString(writer, "tensor"); // Duplicate name
            writer.Write(1u);
            writer.Write(200UL);
            writer.Write(2u);
            writer.Write(100UL);
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Contains("duplicate name", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RejectsTensorDescriptorWithZeroDimension()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
            WriteString(writer, "tensor");
            writer.Write(2u);
            writer.Write(100UL);
            writer.Write(0UL);
            writer.Write(1u);
            writer.Write(0UL);
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Contains("zero-sized dimension", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task RejectsTensorDescriptorWithTruncatedNameBytes()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
            writer.Write(30UL);
            writer.Write(Encoding.ASCII.GetBytes("abcdefghijklmnop"));
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Equal("GGUF string is too large or truncated.", exception.Message);
    }

    [Fact]
    public async Task RejectsTensorDescriptorWithTruncatedNDimensions()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
            WriteString(writer, "abcdefghijklmn");
            writer.Write((ushort)2u); // Only two of four count bytes.
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Equal("GGUF file is truncated.", exception.Message);
    }

    [Fact]
    public async Task RejectsTensorDescriptorWithTruncatedDimension()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
            WriteString(writer, "tensor01");
            writer.Write(1u);
            writer.Write(100u); // Only four of eight dimension bytes.
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Equal("GGUF file is truncated.", exception.Message);
    }

    [Fact]
    public async Task RejectsTensorDescriptorWithTruncatedType()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
            WriteString(writer, "tensor");
            writer.Write(1u);
            writer.Write(100UL);
            // Only write 2 bytes for ggml_type instead of 4
            writer.Write((ushort)1u);
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Equal("GGUF file is truncated.", exception.Message);
    }

    [Fact]
    public async Task RejectsTensorDescriptorWithTruncatedOffset()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
            WriteString(writer, "tensor");
            writer.Write(1u);
            writer.Write(100UL);
            writer.Write(1u);
            // Only write 4 bytes for offset instead of 8
            writer.Write((uint)0UL);
        });

        var exception = await Assert.ThrowsAsync<GgufFormatException>(
            () => new GgufReader().ReadAsync(stream));

        Assert.Equal("GGUF file is truncated.", exception.Message);
    }

    [Fact]
    public void PreservesThreeFieldGgufFileRecord()
    {
        var metadata = new Dictionary<string, GgufMetadataValue>();
        var file = new GgufFile(3, 0, metadata);
        var (version, tensorCount, deconstructedMetadata) = file;

        Assert.Equal(3u, file.Version);
        Assert.Equal(0UL, file.TensorCount);
        Assert.Empty(file.Tensors);
        Assert.Equal(3u, version);
        Assert.Equal(0UL, tensorCount);
        Assert.Same(metadata, deconstructedMetadata);
    }

    [Fact]
    public async Task ReadsValidTensorDescriptorsWithLargeDimensions()
    {
        using var stream = CreateFile(writer =>
        {
            WriteHeader(writer, version: 3, tensorCount: 1, metadataCount: 0);
            WriteString(writer, "tensor_large");
            writer.Write(2u);
            writer.Write(uint.MaxValue + 1UL); // Large dimension > uint.max
            writer.Write(1000UL);
            writer.Write(1u);
            writer.Write(0UL);
        });

        var result = await new GgufReader().ReadAsync(stream);

        Assert.Equal(1UL, result.TensorCount);
        Assert.Equal(1, result.Tensors.Count);

        var tensor = result.Tensors[0];
        Assert.Equal("tensor_large", tensor.Name);
        Assert.Equal(new ulong[] { uint.MaxValue + 1UL, 1000UL }, tensor.Dimensions);
        Assert.Equal(1u, tensor.GgmlType);
        Assert.Equal(0UL, tensor.Offset);
    }

    private static MemoryStream CreateFile(Action<BinaryWriter> write)
    {
        var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            write(writer);
        }

        stream.Position = 0;
        return stream;
    }

    private static void WriteHeader(
        BinaryWriter writer,
        uint version,
        ulong tensorCount,
        ulong metadataCount)
    {
        writer.Write(Encoding.ASCII.GetBytes("GGUF"));
        writer.Write(version);
        writer.Write(tensorCount);
        writer.Write(metadataCount);
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }
}
