using LMSharp.Console;
using LMSharp.Inference;
using LMSharp.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace LMSharp.UnitTests;

[Collection("Phi-4 model execution")]
public sealed class Phi4CudaTests(ITestOutputHelper output)
{
    [Phi4CudaFact]
    public async Task TerminalSwitchesFromCpuToCudaAndBackInTheSameRun()
    {
        using var chat = await ChatRuntime.OpenAsync(new ChatSettings(
            Phi4ModelFactAttribute.ModelPath, 96, null, new GenerationOptions(MaxTokens: 12), InferenceDevice.Cpu));
        using var input = new StringReader("""
            Reply with only the digit. What is 2 + 2?
            /device toggle
            /settings
            Remember this codeword: ORCHID. Reply with only OK.
            /threads 4
            What codeword did I ask you to remember? Reply with only the codeword.
            /device cpu
            /settings
            Reply with only the digit. What is 2 + 2?
            /exit
            """);
        using var transcript = new StringWriter();

        await ChatCommand.RunLoopAsync(chat, input, transcript);

        var text = transcript.ToString();
        Assert.Contains("Model loaded on CUDA (VRAM)", text, StringComparison.Ordinal);
        Assert.Contains("/threads updated. Conversation retained.", text, StringComparison.Ordinal);
        Assert.Contains("Assistant> ORCHID", text, StringComparison.Ordinal);
        Assert.Contains("Model loaded on CPU (RAM). Conversation reset.", text, StringComparison.Ordinal);
        Assert.Equal(2, text.Split("Context: 0/96 tokens used", StringSplitOptions.None).Length - 1);
        Assert.Equal(2, text.Split($"Assistant> 4{Environment.NewLine}", StringSplitOptions.None).Length - 1);
        Assert.Equal(InferenceDevice.Cpu, chat.Settings.Device);
        Assert.Equal(4, chat.Settings.Generation.Threads);
    }

    [Phi4CudaFact]
    public async Task PackedCudaMatchesCpuLogitsAndRetainsConversation()
    {
        var path = Phi4ModelFactAttribute.ModelPath;
        using var cpu = await Phi4ModelLoader.LoadAsync(path, InferenceDevice.Cpu);
        using var gpu = await Phi4ModelLoader.LoadAsync(path, InferenceDevice.Cuda);
        Assert.Equal("CPU (RAM)", cpu.DeviceName);
        Assert.StartsWith("CUDA (VRAM)", gpu.DeviceName);
        output.WriteLine(gpu.DeviceName);

        var input = cpu.Tokenizer.Encode(Phi4Tokenizer.FormatPrompt("What is 2 + 2?"));
        using var cpuSession = cpu.CreateSession(64);
        using var gpuSession = gpu.CreateSession(64);
        AssertClose(cpuSession.Prefill(input), gpuSession.Prefill(input));
        var withCachedToken = gpuSession.Advance(976).ToArray();
        AssertClose(cpuSession.Advance(976), withCachedToken);
        gpuSession.Rewind(0);
        Assert.Equal(withCachedToken, gpuSession.Prefill([.. input, 976]).ToArray());

        var longerInput = gpu.Tokenizer.Encode(Phi4Tokenizer.FormatPrompt(string.Concat(Enumerable.Repeat(" word", 40))));
        Assert.InRange(longerInput.Count, 33, gpuSession.Capacity);
        gpuSession.Rewind(0);
        var longerLogits = gpuSession.Prefill(longerInput).ToArray();
        gpuSession.Rewind(0);
        gpuSession.Prefill(longerInput.Take(16).ToArray());
        Assert.Equal(longerLogits, gpuSession.Prefill(longerInput.Skip(16).ToArray()).ToArray());

        var single = Phi4Generator.Generate(gpu, "What is 2 + 2?", new GenerationOptions(MaxTokens: 12));
        Assert.Equal([976, 6052, 316, 220, 17, 659, 220, 17, 382, 220, 19, 13], single.TokenIds);

        using var conversation = new Phi4ChatSession(gpu, contextLength: 128);
        Assert.Equal("OK", conversation.Reply("Remember this codeword: ORCHID. Reply with only OK.").Text);
        Assert.Equal("ORCHID",
            conversation.Reply("What codeword did I ask you to remember? Reply with only the codeword.").Text);
        Assert.Equal(43, conversation.ContextTokensUsed);
        conversation.Reset();
        Assert.Equal(0, conversation.ContextTokensUsed);
    }

    private void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maximumError = 0;
        for (var index = 0; index < expected.Length; index++)
        {
            maximumError = MathF.Max(maximumError, MathF.Abs(expected[index] - actual[index]));
        }

        output.WriteLine($"Maximum absolute logit difference: {maximumError}");
        Assert.InRange(maximumError, 0, 0.001f);
    }

    private sealed class Phi4CudaFactAttribute : FactAttribute
    {
        public Phi4CudaFactAttribute()
        {
            if (Environment.GetEnvironmentVariable("LMSHARP_TEST_CUDA") != "1" ||
                string.IsNullOrWhiteSpace(Phi4ModelFactAttribute.ModelPath))
            {
                Skip = "Set LMSHARP_TEST_CUDA=1 and LMSHARP_PHI4_GGUF to run CUDA integration tests.";
            }
        }
    }
}
