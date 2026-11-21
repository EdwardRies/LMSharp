using LMSharp.Gguf;
using Xunit;

namespace LMSharp.UnitTests;

public sealed class Phi4TensorIntegrationTests
{
    [Phi4ModelFact]
    public async Task ReadsAllFiveStorageTypesAtReferenceOffsets()
    {
        using var file = await new GgufReader().OpenTensorFileAsync(Phi4ModelFactAttribute.ModelPath);
        Assert.Equal("phi3", file.Info.Architecture);
        Assert.Equal(0x7DE720, file.DataOffset);
        Assert.Equal(196, file.Tensors.Count);
        Assert.Equal(
            new[] { (0u, 67), (8u, 1), (12u, 80), (13u, 32), (14u, 16) },
            file.Tensors.Values.GroupBy(tensor => tensor.Descriptor.GgmlType)
                .OrderBy(group => group.Key).Select(group => (group.Key, group.Count())));

        // Reference data from Python gguf 0.19.0 for the artifact identified in Model-Compatibility.md.
        (string Name, long Offset, long Length, string Prefix)[] samples =
        [
            ("output_norm.weight", 0x7DE720, 0x3000, "0000373F00008A3F00005D3F0000823F0000543F0000773F00008D3F00006B3F"),
            ("token_embd.weight", 0x7E18A0, 0x26EC2000, "281CD6081AD8DD11F92602EA285B63E3F20FE52EB415F8DF7F2BD73044D0DA13"),
            ("blk.0.attn_output.weight", 0x276A68A0, 0x510000, "45055F10FFE6B3E4F5DAE6EF23F4274624158369757946E61325756E62552275"),
            ("blk.0.attn_qkv.weight", 0x27BB68A0, 0xA50000, "93038F13F5FFFDF9EEFBEAAAC8FF55C7641359F748613647257731023D4D1472"),
            ("blk.0.ffn_down.weight", 0x286068A0, 0x13B0000, "C119C2EEDBFF6B2055903847133B09D6E0C61C17DD2BC211901B1DF760ED12C3"),
            ("blk.31.ffn_up.weight", 0x9BB638A0, 0x1B00000, "360C8317A7FFB1EFEDB6BFEF2BD24C5480753B69758C997D334D565A2BF2584A"),
        ];

        var prefix = new byte[32];
        foreach (var sample in samples)
        {
            var range = file.Tensors[sample.Name];
            Assert.Equal(sample.Offset, range.FileOffset);
            Assert.Equal(sample.Length, range.ByteLength);
            await file.ReadTensorAsync(sample.Name, 0, prefix);
            Assert.Equal(sample.Prefix, Convert.ToHexString(prefix));
        }
    }
}
