using System.Buffers.Binary;
using System.Security.Cryptography;
using LMSharp.Gguf;
using LMSharp.Tensors;
using Xunit;

namespace LMSharp.UnitTests;

public sealed class CpuKernelsTests
{
    [Fact]
    public void MultipliesPackedRowsWithSignedQuantsAndDifferentScales()
    {
        var bytes = new byte[68];
        BinaryPrimitives.WriteHalfLittleEndian(bytes, (Half)0.5f);
        BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan(34), (Half)2f);
        bytes[2] = 8;
        bytes[33] = unchecked((byte)-4);
        bytes[36] = unchecked((byte)-5);
        bytes[67] = 3;
        var matrix = new PackedMatrix(GgmlStorageType.Q8_0, 2, 32, bytes);
        var input = new float[32];
        input[0] = 2;
        input[31] = 3;
        var output = new float[2];

        matrix.Multiply(input, output, new ParallelOptions());

        Assert.Equal(new float[] { 2, -2 }, output);
        matrix.ReadRow(1, input);
        Assert.Equal(-10, input[0]);
        Assert.Equal(6, input[31]);
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix.ReadRow(2, input));
        Assert.Throws<ArgumentException>(() => matrix.Multiply(input, input, new ParallelOptions()));
    }

    [Fact]
    public void RejectsPartialBlocks()
    {
        Assert.Throws<ArgumentException>(() =>
            GgmlDequantizer.Decode(GgmlStorageType.Q4_K, new byte[144], new float[255]));
        Assert.Throws<ArgumentException>(() =>
            GgmlDequantizer.Decode(GgmlStorageType.Q6_K, new byte[209], new float[256]));
        Assert.Throws<ArgumentException>(() => new PackedMatrix(GgmlStorageType.Q8_0, 1, 31, new byte[34]));
    }

    [Fact]
    public void TensorPrimitivesMatchFloatReference()
    {
        var left = Enumerable.Range(1, 19).Select(i => (float)i).ToArray();
        Assert.Equal(2470, CpuKernels.Dot(left, left));

        var result = new float[3];
        CpuKernels.RmsNorm([3, 4, 0], [2, 0.5f, 3], 0.00001f, result);
        var rms = Math.Sqrt(25.0 / 3 + 0.00001);
        Assert.InRange(Math.Abs(result[0] - 6 / rms), 0, 1e-6);
        Assert.InRange(Math.Abs(result[1] - 2 / rms), 0, 1e-6);
        Assert.Equal(0, result[2]);

        CpuKernels.SwiGlu([-2, 0, 2], [2, 100, 3], result);
        Assert.InRange(Math.Abs(result[0] - -4 / (1 + Math.Exp(2))), 0, 1e-6);
        Assert.Equal(0, result[1]);
        Assert.InRange(Math.Abs(result[2] - 6 / (1 + Math.Exp(-2))), 0, 1e-6);

        float[] scores = [999, 1000, 1001];
        CpuKernels.SoftmaxInPlace(scores);
        Assert.InRange(scores.Sum(), 0.999999f, 1.000001f);
        Assert.InRange(scores[2], 0.6652408f, 0.6652411f);
    }

    [Phi4ModelFact]
    public async Task DequantizesRealRowsAndMatchesReferenceProducts()
    {
        using var file = await new GgufReader().OpenTensorFileAsync(Phi4ModelFactAttribute.ModelPath);
        // gguf 0.19.0 dequantization, first row serialized as little-endian float32,
        // and a float64-accumulated dot product with a repeating input vector.
        (string Name, string Hash, double Dot)[] rows =
        [
            ("output_norm.weight", "58FA78F8A04D0C36193DDAD29F6183FF64D49FB6A11B0F4D3AD0846264C3CEAC", -244.75442504882812),
            ("token_embd.weight", "0E1441C4C0B5D9F5B6EC7FA0AD47593CA654F8560180232D96BD39F77EC832FC", 13.21484375),
            ("blk.0.attn_output.weight", "32E626525CC918B3E66745590DFCCE2DC7A7A67E94BD8C32AAD4939BFBE3B532", -0.9189023673534393),
            ("blk.0.attn_qkv.weight", "A9983682E8938AEB1E5E0DF1BC5328947F252C9E1F04CD66B439CAA380893390", -1.1584226191043854),
            ("blk.0.ffn_down.weight", "E66E093398E560531A038F8D473138121F60BD6E24F6E2DC72F30731035F58BC", -5.469530209898949),
        ];

        foreach (var (name, hash, dot) in rows)
        {
            var tensor = file.Tensors[name];
            var type = (GgmlStorageType)tensor.Descriptor.GgmlType;
            var columns = (int)tensor.Descriptor.Dimensions[0];
            var layout = GgmlBlockLayout.Get(type);
            var bytes = new byte[columns / layout.ElementCount * layout.ByteCount];
            await file.ReadTensorAsync(name, 0, bytes);
            var matrix = new PackedMatrix(type, rows: 1, columns, bytes);
            var decoded = new float[columns];
            matrix.ReadRow(0, decoded);

            var decodedBytes = new byte[columns * sizeof(float)];
            for (var i = 0; i < columns; i++)
            {
                BinaryPrimitives.WriteSingleLittleEndian(decodedBytes.AsSpan(i * sizeof(float)), decoded[i]);
            }

            Assert.Equal(hash, Convert.ToHexString(SHA256.HashData(decodedBytes)));
            float[] cycle = [-2, 0.5f, 1.25f];
            var input = Enumerable.Range(0, columns).Select(i => cycle[i % cycle.Length]).ToArray();
            var product = new float[1];
            matrix.Multiply(input, product, new ParallelOptions());
            Assert.InRange(Math.Abs(product[0] - dot), 0, 0.0001);
        }
    }
}
