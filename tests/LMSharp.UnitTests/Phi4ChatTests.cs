using LMSharp.Console;
using LMSharp.Inference;
using LMSharp.Tokenization;
using Xunit;

namespace LMSharp.UnitTests;

[Collection("Phi-4 model execution")]
public sealed class Phi4ChatTests
{
    [Phi4ModelFact]
    public async Task TerminalSettingsKeepConversationUntilAResettingChange()
    {
        using var chat = await ChatRuntime.OpenAsync(new ChatSettings(
            Phi4ModelFactAttribute.ModelPath, 96, null, new GenerationOptions(), InferenceDevice.Cpu));
        using var input = new StringReader($"""
            /help
            /max-tokens 1
            Remember this codeword: ORCHID. Reply with only OK.
            /max-tokens 12
            /temperature 0.5
            /temperature 0
            /seed -21
            /max-tokens 0
            /model "{Phi4ModelFactAttribute.ModelPath}.missing"
            /context 4097
            /settings
            What codeword did I ask you to remember? Reply with only the codeword.
            /context 128
            /threads 2
            /system Answer briefly.
            /settings
            /system
            /reset
            /exit
            This line comes after /exit.
            """);
        using var output = new StringWriter();

        await ChatCommand.RunLoopAsync(chat, input, output);

        var transcript = output.ToString();
        Assert.Contains("/device cpu|cuda|toggle", transcript, StringComparison.Ordinal);
        Assert.Contains("Assistant> OK", transcript, StringComparison.Ordinal);
        Assert.Contains("Reply reached its token limit.", transcript, StringComparison.Ordinal);
        Assert.Contains("Max reply tokens: 12", transcript, StringComparison.Ordinal);
        Assert.Contains("Seed: -21", transcript, StringComparison.Ordinal);
        Assert.Contains("Command failed: /max-tokens must be a positive integer.", transcript, StringComparison.Ordinal);
        Assert.Contains(".gguf.missing", transcript, StringComparison.Ordinal);
        Assert.Contains("Assistant> ORCHID", transcript, StringComparison.Ordinal);
        Assert.Contains("Context used: 43/96 tokens.", transcript, StringComparison.Ordinal);
        Assert.Contains("/context updated. Conversation reset.", transcript, StringComparison.Ordinal);
        Assert.Contains("/threads updated. Conversation reset.", transcript, StringComparison.Ordinal);
        Assert.Contains("System prompt: Answer briefly.", transcript, StringComparison.Ordinal);
        Assert.Contains("Conversation cleared.", transcript, StringComparison.Ordinal);
        Assert.Equal(Phi4ModelFactAttribute.ModelPath, chat.Settings.ModelPath);
        Assert.Equal(128, chat.Conversation.ContextLength);
        Assert.Equal(0, chat.Conversation.ContextTokensUsed);
        Assert.Null(chat.Settings.SystemPrompt);
        Assert.Equal(2, chat.Settings.Generation.Threads);
        Assert.Equal("This line comes after /exit.", input.ReadLine());
    }

    [Phi4ModelFact]
    public async Task CachedFollowUpMatchesReplayedHistoryAfterACappedReply()
    {
        const string system = "Keep answers brief.";
        const string firstQuestion = "Reply with only the digit. What is 2 + 2?";
        const string followUp = "Now repeat that answer.";
        var model = await Phi4Model.LoadAsync(Phi4ModelFactAttribute.ModelPath);
        var chat = new Phi4ChatSession(model, contextLength: 96, new GenerationOptions(MaxTokens: 1), system);
        var first = chat.Reply(firstQuestion);
        Assert.Equal(GenerationStopReason.TokenLimit, first.StopReason);
        var firstPrompt = model.Tokenizer.Encode(Phi4Tokenizer.FormatPrompt(firstQuestion, system));
        var used = firstPrompt.Count + first.TokenIds.Count + 1;
        Assert.Equal(used, chat.ContextTokensUsed);

        Assert.Throws<ChatContextFullException>(() => chat.Reply(string.Concat(Enumerable.Repeat(" long", 100))));
        Assert.Throws<OperationCanceledException>(() => chat.Reply(followUp, new CancellationToken(true)));
        Assert.Equal(used, chat.ContextTokensUsed);

        var nextPrompt = model.Tokenizer.Encode(Phi4Tokenizer.FormatPrompt(followUp));
        var replay = model.CreateSession(contextLength: chat.ContextLength);
        var logits = replay.Prefill([.. firstPrompt, .. first.TokenIds, model.Tokenizer.EndOfTurnId, .. nextPrompt]);
        var expected = new TokenSampler(0, seed: null).Choose(logits);

        var answer = chat.Reply(followUp);

        Assert.Equal(expected, Assert.Single(answer.TokenIds));
        var closingMarker = answer.StopReason == GenerationStopReason.TokenLimit ? 1 : 0;
        Assert.Equal(used + nextPrompt.Count + answer.TokenIds.Count + closingMarker, chat.ContextTokensUsed);
    }

    [Fact]
    public void ParsesChatOptionsWithoutRequiringAPresetPrompt()
    {
        var defaults = ChatCommand.Parse(["chat", "--model", "model.gguf"]);
        Assert.Equal(new ChatSettings("model.gguf", 1024, null, new GenerationOptions(), InferenceDevice.Cpu), defaults);
        var explicitSettings = ChatCommand.Parse(
            ["chat", "--context", "512", "--model", "model.gguf", "--system", "Be brief.", "--max-tokens", "24",
             "--device", "cuda"]);
        Assert.Equal(new ChatSettings("model.gguf", 512, "Be brief.", new GenerationOptions(MaxTokens: 24),
            InferenceDevice.Cuda), explicitSettings);
    }

    [Theory]
    [InlineData("--prompt", "Hello")]
    [InlineData("--context", "0")]
    [InlineData("--context", "abc")]
    [InlineData("--device", "vulkan")]
    public void RejectsInvalidChatOptions(string flag, string value)
    {
        Assert.Throws<ArgumentException>(() => ChatCommand.Parse(["chat", "--model", "model.gguf", flag, value]));
    }
}
