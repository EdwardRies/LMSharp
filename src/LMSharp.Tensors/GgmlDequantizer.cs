using System.Buffers.Binary;
using System.Runtime.Intrinsics;
using LMSharp.Gguf;

namespace LMSharp.Tensors;

public static class GgmlDequantizer
{
    public static void Decode(GgmlStorageType type, ReadOnlySpan<byte> source, Span<float> values)
    {
        var layout = GgmlBlockLayout.Get(type);
        if (values.Length % layout.ElementCount != 0 ||
            source.Length != (long)(values.Length / layout.ElementCount) * layout.ByteCount)
        {
            throw new ArgumentException("Source and destination must describe the same number of full GGML blocks.");
        }

        while (!values.IsEmpty)
        {
            DecodeBlock(type, source[..layout.ByteCount], values[..layout.ElementCount]);
            source = source[layout.ByteCount..];
            values = values[layout.ElementCount..];
        }
    }

    internal static void DecodeBlock(GgmlStorageType type, ReadOnlySpan<byte> block, Span<float> values)
    {
        switch (type)
        {
            case GgmlStorageType.F32:
                values[0] = BinaryPrimitives.ReadSingleLittleEndian(block);
                break;
            case GgmlStorageType.Q8_0:
                DecodeQ8(block, values);
                break;
            case GgmlStorageType.Q4_K:
                DecodeKWithMin(block, values, fiveBit: false);
                break;
            case GgmlStorageType.Q5_K:
                DecodeKWithMin(block, values, fiveBit: true);
                break;
            case GgmlStorageType.Q6_K:
                DecodeQ6(block, values);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(type));
        }
    }

    private static float ReadHalf(ReadOnlySpan<byte> bytes) => (float)BinaryPrimitives.ReadHalfLittleEndian(bytes);

    private static void DecodeQ8(ReadOnlySpan<byte> block, Span<float> values)
    {
        var scale = ReadHalf(block);
        var quants = block[2..];
        for (var i = 0; i < 32; i++)
        {
            values[i] = scale * (sbyte)quants[i];
        }
    }

    private static void DecodeKWithMin(ReadOnlySpan<byte> block, Span<float> values, bool fiveBit)
    {
        var blockScale = ReadHalf(block);
        var blockMin = ReadHalf(block[2..]);
        var scales = block[4..16];
        var upperBits = fiveBit ? block[16..48] : default;
        var lowerBits = block[(fiveBit ? 48 : 16)..];

        // Eight groups of 32. Adjacent groups share low and high nibbles.
        for (var group = 0; group < 8; group++)
        {
            var (scale, min) = ReadScaleAndMin(scales, group);
            var multiplier = blockScale * scale;
            var offset = blockMin * min;
            var quants = lowerBits.Slice((group / 2) * 32, 32);
            var output = values.Slice(group * 32, 32);
            var shift = (group % 2) * 4;
            if (Vector256.IsHardwareAccelerated)
            {
                DecodeKGroupVectorized(quants, upperBits, group, shift, multiplier, offset, output);
            }
            else if (fiveBit)
            {
                var mask = 1 << group;
                for (var i = 0; i < 32; i++)
                {
                    var quant = ((quants[i] >> shift) & 15) + ((upperBits[i] & mask) == 0 ? 0 : 16);
                    output[i] = multiplier * quant - offset;
                }
            }
            else
            {
                for (var i = 0; i < 32; i++)
                {
                    output[i] = multiplier * ((quants[i] >> shift) & 15) - offset;
                }
            }
        }
    }

    private static void DecodeKGroupVectorized(
        ReadOnlySpan<byte> lowBits, ReadOnlySpan<byte> highBits, int group, int shift,
        float multiplier, float offset, Span<float> output)
    {
        var packed = Vector256.Create(lowBits);
        var quants = Vector256.ShiftRightLogical(packed, shift) & Vector256.Create((byte)15);
        if (!highBits.IsEmpty)
        {
            var high = Vector256.Create(highBits) & Vector256.Create((byte)(1 << group));
            var zeroHigh = Vector256.Equals(high, Vector256<byte>.Zero);
            quants |= ~zeroHigh & Vector256.Create((byte)16);
        }

        var scale = Vector256.Create(multiplier);
        var min = Vector256.Create(offset);
        Store16(Vector256.WidenLower(quants), scale, min, output);
        Store16(Vector256.WidenUpper(quants), scale, min, output[16..]);
    }

    private static void Store16(
        Vector256<ushort> quants, Vector256<float> scale, Vector256<float> min, Span<float> output)
    {
        var first = Vector256.ConvertToSingle(Vector256.WidenLower(quants));
        var second = Vector256.ConvertToSingle(Vector256.WidenUpper(quants));
        (first * scale - min).CopyTo(output);
        (second * scale - min).CopyTo(output[8..]);
    }

    private static (int Scale, int Min) ReadScaleAndMin(ReadOnlySpan<byte> scales, int group) =>
        group < 4
            ? (scales[group] & 63, scales[group + 4] & 63)
            : ((scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
               (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4));

    private static void DecodeQ6(ReadOnlySpan<byte> block, Span<float> values)
    {
        var blockScale = ReadHalf(block[208..]);
        var scales = block[192..208];
        var lowerBits = block[..128];
        var upperBits = block[128..192];
        for (var half = 0; half < 2; half++)
        {
            for (var group = 0; group < 4; group++)
            {
                var low = lowerBits.Slice(half * 64 + (group % 2) * 32, 32);
                var high = upperBits.Slice(half * 32, 32);
                var output = values.Slice(half * 128 + group * 32, 32);
                var scaleIndex = half * 8 + group * 2;
                var lowShift = group >= 2 ? 4 : 0;
                var highShift = group * 2;
                for (var part = 0; part < 2; part++)
                {
                    var multiplier = blockScale * (sbyte)scales[scaleIndex + part];
                    for (var i = part * 16; i < (part + 1) * 16; i++)
                    {
                        var quant = ((low[i] >> lowShift) & 15) + (((high[i] >> highShift) & 3) << 4) - 32;
                        output[i] = multiplier * quant;
                    }
                }
            }
        }
    }
}
