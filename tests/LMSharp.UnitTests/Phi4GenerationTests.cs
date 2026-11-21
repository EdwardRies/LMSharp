using LMSharp.Console;
using LMSharp.Inference;
using Xunit;

namespace LMSharp.UnitTests;

[Collection("Phi-4 model execution")]
public sealed class Phi4GenerationTests
{
    [Phi4ModelFact]
    public async Task MatchesReferenceGreedyResponseAndStopsOnLimitOrEndOfTurn()
    {
        var model = await Phi4Model.LoadAsync(Phi4ModelFactAttribute.ModelPath);
        Assert.Throws<ArgumentException>(() =>
            Phi4Generator.Generate(model, "Hello", new GenerationOptions(MaxTokens: model.MaximumContextLength)));

        // llama.cpp b12608 CPU, temperature 0, exact Phi-4 chat prompt IDs.
        var bounded = Phi4Generator.Generate(model, "What is 2 + 2?", new GenerationOptions(MaxTokens: 12));
        Assert.Equal(GenerationStopReason.TokenLimit, bounded.StopReason);
        Assert.Equal(11, bounded.PromptTokens);
        Assert.Equal([976, 6052, 316, 220, 17, 659, 220, 17, 382, 220, 19, 13], bounded.TokenIds);
        Assert.Equal("The answer to 2 + 2 is 4.", bounded.Text);

        var ended = Phi4Generator.Generate(model, "Reply with only the digit. What is 2 + 2?",
            new GenerationOptions(MaxTokens: 12));
        Assert.Equal(GenerationStopReason.EndOfSequence, ended.StopReason);
        Assert.Equal(17, ended.PromptTokens);
        Assert.Equal([19, 200020], ended.TokenIds);
        Assert.Equal("4", ended.Text);
    }

    [Fact]
    public void GreedySelectsFirstMaximumAndRejectsInvalidScores()
    {
        var sampler = new TokenSampler(temperature: 0, seed: null);
        Assert.Equal(1, sampler.Choose([-10, 5, 5, 4]));
        Assert.Throws<ArgumentException>(() => sampler.Choose([]));
        Assert.Throws<InvalidOperationException>(() => sampler.Choose([0, float.NaN]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TokenSampler(-1, seed: null));
    }

    [Fact]
    public void StochasticSamplingRepeatsWithTheSameSeed()
    {
        var first = new TokenSampler(temperature: 0.8f, seed: 123);
        var second = new TokenSampler(temperature: 0.8f, seed: 123);
        float[] scores = [5000f, 5000.5f, 4999.5f, -5000f];
        var samples = Enumerable.Range(0, 24).Select(_ => first.Choose(scores)).ToArray();
        Assert.Equal(samples, Enumerable.Range(0, 24).Select(_ => second.Choose(scores)));
        Assert.True(samples.Distinct().Count() > 1);
        Assert.DoesNotContain(3, samples);
    }

    [Fact]
    public void ParsesConsoleOptionsAndDefaults()
    {
        var defaults = GenerateCommand.Parse(["generate", "--model", "model.gguf", "--prompt", "Hello"]);
        Assert.Equal(("model.gguf", "Hello", (string?)null, new GenerationOptions(), InferenceDevice.Cpu), defaults);
        var explicitOptions = GenerateCommand.Parse(
            ["generate", "--prompt", "Hi", "--system", "Be brief", "--model", "model.gguf",
             "--max-tokens", "12", "--threads", "8", "--temperature", "0.5", "--seed", "-7", "--device", "cuda"]);
        Assert.Equal(("model.gguf", "Hi", "Be brief", new GenerationOptions(12, 0.5f, -7, 8), InferenceDevice.Cuda), explicitOptions);
    }

    [Theory]
    [InlineData("--unknown", "x")]
    [InlineData("--max-tokens", "0")]
    [InlineData("--threads", "-1")]
    [InlineData("--temperature", "NaN")]
    [InlineData("--temperature", "0,5")]
    [InlineData("--seed", "one")]
    [InlineData("--device", "auto")]
    [InlineData("--prompt", "Duplicate prompt")]
    public void RejectsInvalidConsoleOptions(string flag, string value)
    {
        Assert.Throws<ArgumentException>(() =>
            GenerateCommand.Parse(["generate", "--model", "model.gguf", "--prompt", "Hello", flag, value]));
    }
}
