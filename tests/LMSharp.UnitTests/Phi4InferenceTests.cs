using LMSharp.Inference;
using LMSharp.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace LMSharp.UnitTests;

[Collection("Phi-4 model execution")]
public sealed class Phi4InferenceTests(ITestOutputHelper output)
{
    [Fact]
    public void RotatesOnlyTheConfiguredPrefixOfEachHead()
    {
        var rotary = new Phi4RotaryEmbedding(frequencyBase: 16, scale: 2, [1, 2]);
        rotary.SetPosition(1);
        float[] heads = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6];
        rotary.RotateHeads(heads, headSize: 6);

        Assert.InRange(Math.Abs(heads[0] - 2 * (Math.Cos(1) - 3 * Math.Sin(1))), 0, 1e-6);
        Assert.InRange(Math.Abs(heads[1] - 2 * (2 * Math.Cos(0.125) - 4 * Math.Sin(0.125))), 0, 1e-6);
        Assert.InRange(Math.Abs(heads[2] - 2 * (3 * Math.Cos(1) + Math.Sin(1))), 0, 1e-6);
        Assert.InRange(Math.Abs(heads[3] - 2 * (4 * Math.Cos(0.125) + 2 * Math.Sin(0.125))), 0, 1e-6);
        Assert.Equal(5, heads[4]);
        Assert.Equal(6, heads[5]);
        for (var i = 0; i < 6; i++)
        {
            Assert.Equal(-heads[i], heads[i + 6]);
        }
    }

    [Phi4ModelFact]
    public async Task MatchesFirstLayerAndFloat32ReferenceLogits()
    {
        var model = await Phi4Model.LoadAsync(Phi4ModelFactAttribute.ModelPath);
        var prompt = model.Tokenizer.Encode(Phi4Tokenizer.FormatPrompt("What is 2 + 2?"));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.CreateSession(4097));
        var session = model.CreateSession(contextLength: prompt.Count + 1);
        Assert.Throws<ArgumentOutOfRangeException>(() => session.Prefill([200021, 200064]));
        Assert.ThrowsAny<OperationCanceledException>(() => session.Prefill(prompt, new CancellationToken(true)));
        Assert.Equal(0, session.Position);

        // NumPy float32 layer math with Python gguf 0.19.0 dequantized weights.
        int[] sampleIndices = [0, 1, 2, 47, 48, 95, 96, 127, 128, 513, 1234, 2047, 3071];
        float[][] expectedLayer =
        [
            [0.00998721f, 0.07515875f, 0.06839792f, -0.07277110f, 0.09412579f, 0.05513910f,
             0.06294593f, -0.12616602f, -0.03881917f, -0.11495030f, -0.09132392f, -0.07993737f, -0.03390058f],
            [0.08941980f, -0.09847217f, -0.12206273f, -0.02609869f, -0.15754208f, -0.06896468f,
             -0.15390557f, 0.24466376f, -0.08403379f, -0.19057173f, -0.21397886f, 0.04812193f, -0.03263332f],
        ];
        var observedLayer = new List<float[]>();
        var next = session.Prefill(prompt, default, (position, layer, state) =>
        {
            if (layer == 0 && position < expectedLayer.Length)
            {
                observedLayer.Add(sampleIndices.Select(index => state.Span[index]).ToArray());
            }
        }).ToArray();
        Assert.Equal(expectedLayer.Length, observedLayer.Count);
        for (var position = 0; position < expectedLayer.Length; position++)
        {
            for (var sample = 0; sample < sampleIndices.Length; sample++)
            {
                Assert.InRange(Math.Abs(observedLayer[position][sample] - expectedLayer[position][sample]), 0, 0.00002f);
            }
        }

        // Full dequantized float32 oracle in tests/reference/phi4_float_reference.py.
        // llama.cpp b12608 also selects 976, with wider score differences from its quantized path.
        Assert.True(Array.TrueForAll(next, float.IsFinite));
        Assert.Equal(976, Array.IndexOf(next, next.Max()));
        (int Token, float ExpectedLogit)[] expectedLogits =
            [(976, 48.660107f), (17, 48.582932f), (13866, 44.388966f), (7924, 44.117317f), (702, 43.989403f)];
        foreach (var (token, expectedLogit) in expectedLogits)
        {
            output.WriteLine($"Token {token}: logit {next[token]}, reference {expectedLogit}");
            Assert.InRange(Math.Abs(next[token] - expectedLogit), 0, 0.001f);
        }

        var advanced = session.Advance(976).ToArray();
        Assert.Equal(6052, Array.IndexOf(advanced, advanced.Max()));
        Assert.Throws<InvalidOperationException>(() => session.Advance(6052));

        session.Reset();
        var replay = session.Prefill([.. prompt, 976]).ToArray();
        Assert.Equal(advanced, replay);
        Assert.Equal(session.Capacity, session.Position);
    }
}
