using System.Text;
using LMSharp.Gguf;
using Xunit;

namespace LMSharp.UnitTests;

public sealed class GgufTensorFileTests
{
    [Fact]
    public Task ReadsBoundedSlicesWithCustomAlignmentAndUnsortedDescriptors() => WithFileAsync(writer =>
    {
        WriteHeader(writer, tensorCount: 2, alignment: 48);
        WriteTensor(writer, "second", type: 0, offset: 48, 1);
        WriteTensor(writer, "first", type: 0, offset: 0, 2);
        PadToAlignment(writer, 48);
        writer.Write(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        PadToAlignment(writer, 48);
        writer.Write(new byte[] { 9, 10, 11, 12 });
    }, async path =>
    {
        using var file = await new GgufReader().OpenTensorFileAsync(path);
        Assert.Equal(144, file.DataOffset);
        Assert.Equal(192, file.Tensors["second"].FileOffset);
        Assert.Equal(8, file.Tensors["first"].ByteLength);

        var first = new byte[3];
        var second = new byte[4];
        await Task.WhenAll(
            file.ReadTensorAsync("first", 5, first).AsTask(),
            file.ReadTensorAsync("second", 0, second).AsTask());
        Assert.Equal(new byte[] { 6, 7, 8 }, first);
        Assert.Equal(new byte[] { 9, 10, 11, 12 }, second);
        await file.ReadTensorAsync("first", 8, Memory<byte>.Empty);

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => file.ReadTensorAsync("first", 7, first).AsTask());
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => file.ReadTensorAsync("first", -1, first).AsTask());
        await Assert.ThrowsAsync<KeyNotFoundException>(
            () => file.ReadTensorAsync("missing", 0, first).AsTask());

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => file.ReadTensorAsync("first", 0, first, new CancellationToken(true)).AsTask());
        file.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(
            () => file.ReadTensorAsync("first", 0, first).AsTask());
    });

    [Theory]
    [InlineData(0u, 1, 4)]
    [InlineData(8u, 32, 34)]
    [InlineData(12u, 256, 144)]
    [InlineData(13u, 256, 176)]
    [InlineData(14u, 256, 210)]
    public Task SizesQuantizedRowsAndUsesDefaultAlignment(uint type, int valuesPerBlock, int bytesPerBlock) =>
        WithFileAsync(writer =>
        {
            WriteHeader(writer, tensorCount: 1);
            WriteTensor(writer, "weights", type, offset: 0, (ulong)valuesPerBlock * 2, 3);
            PadToAlignment(writer, 32);
            writer.Write(new byte[bytesPerBlock * 6]);
        }, async path =>
        {
            using var file = await new GgufReader().OpenTensorFileAsync(path);
            Assert.Equal(96, file.DataOffset);
            Assert.Equal(bytesPerBlock * 6, file.Tensors["weights"].ByteLength);
        });

    [Theory]
    [InlineData(0u)]
    [InlineData(12u)]
    public Task RejectsInvalidAlignment(uint alignment) => RejectsAsync(writer =>
    {
        WriteHeader(writer, tensorCount: 1, alignment: alignment);
        WriteTensor(writer, "weights", type: 0, offset: 0, 1);
    }, "general.alignment");

    [Fact]
    public Task RejectsTruncatedPadding() => RejectsAsync(writer =>
    {
        WriteHeader(writer, tensorCount: 1);
        WriteTensor(writer, "weights", type: 0, offset: 0, 1);
    }, "aligned tensor data section");

    [Theory]
    [InlineData(0u, 1UL, 2UL, 1UL, 64, "unaligned")]
    [InlineData(0u, 1UL, 2UL, ulong.MaxValue - 31, 64, "beyond the end")]
    [InlineData(0u, 1UL, 2UL, 0UL, 7, "beyond the end")]
    [InlineData(8u, 16UL, 2UL, 0UL, 64, "row dimension")]
    [InlineData(0u, 4UL, ulong.MaxValue, 0UL, 64, "overflows")]
    [InlineData(1u, 1UL, 1UL, 0UL, 64, "unsupported GGML type 1")]
    public Task RejectsInvalidTensorRanges(
        uint type, ulong rowLength, ulong rows, ulong offset, int dataLength, string expectedError) =>
        RejectsAsync(writer =>
        {
            WriteHeader(writer, tensorCount: 1);
            WriteTensor(writer, "weights", type, offset, rowLength, rows);
            PadToAlignment(writer, 32);
            writer.Write(new byte[dataLength]);
        }, expectedError);

    [Fact]
    public Task RejectsOverlappingTensors() => RejectsAsync(writer =>
    {
        WriteHeader(writer, tensorCount: 2);
        WriteTensor(writer, "second", type: 0, offset: 32, 8);
        WriteTensor(writer, "first", type: 0, offset: 0, 16);
        PadToAlignment(writer, 32);
        writer.Write(new byte[64]);
    }, "overlaps");

    [Fact]
    public Task RejectsMetadataOnlyFilesForTensorAccess() => RejectsAsync(
        writer => WriteHeader(writer, tensorCount: 0), "no tensor weights");

    private static Task RejectsAsync(Action<BinaryWriter> write, string expectedError) =>
        WithFileAsync(write, async path =>
        {
            var error = await Assert.ThrowsAsync<GgufFormatException>(async () =>
            {
                using var file = await new GgufReader().OpenTensorFileAsync(path);
            });
            Assert.Contains(expectedError, error.Message, StringComparison.Ordinal);
        });

    private static async Task WithFileAsync(Action<BinaryWriter> write, Func<string, Task> test)
    {
        var path = Path.GetTempFileName();
        try
        {
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                write(writer);
            }

            await test(path);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static void WriteHeader(BinaryWriter writer, ulong tensorCount, uint? alignment = null)
    {
        writer.Write("GGUF"u8);
        writer.Write(3u);
        writer.Write(tensorCount);
        writer.Write(alignment.HasValue ? 1UL : 0UL);
        if (alignment is uint value)
        {
            WriteString(writer, "general.alignment");
            writer.Write((uint)GgufValueType.UInt32);
            writer.Write(value);
        }
    }

    private static void WriteTensor(BinaryWriter writer, string name, uint type, ulong offset, params ulong[] dimensions)
    {
        WriteString(writer, name);
        writer.Write((uint)dimensions.Length);
        foreach (var dimension in dimensions)
        {
            writer.Write(dimension);
        }

        writer.Write(type);
        writer.Write(offset);
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static void PadToAlignment(BinaryWriter writer, int alignment)
    {
        var padding = (alignment - writer.BaseStream.Position % alignment) % alignment;
        writer.Write(new byte[padding]);
    }
}
