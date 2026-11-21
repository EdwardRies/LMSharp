using LMSharp.Console;
using LMSharp.Inference;
using Xunit;

namespace LMSharp.UnitTests;

public sealed class ChatSlashCommandTests
{
    private static readonly ChatSettings Defaults =
        new("model.gguf", 1024, null, new GenerationOptions(), InferenceDevice.Cpu);

    [Fact]
    public void ModelCommandPreservesSpacesAndRemovesOptionalQuotes()
    {
        var (command, value) = ChatSlashCommands.Split("/MODEL\t\"a folder\\phi model.gguf\"");
        var result = ChatSlashCommands.Change(Defaults, command, value);
        Assert.Equal("a folder\\phi model.gguf", result.ModelPath);
        Assert.Equal(result, ChatSlashCommands.Change(Defaults, command, "a folder\\phi model.gguf"));
    }

    [Fact]
    public void TogglesRamAndVramAndAcceptsExplicitDevices()
    {
        var cuda = ChatSlashCommands.Change(Defaults, "/device", "toggle");
        Assert.Equal(InferenceDevice.Cuda, cuda.Device);
        Assert.Equal(Defaults, ChatSlashCommands.Change(cuda, "/device", "toggle"));
        Assert.Equal(cuda, ChatSlashCommands.Change(Defaults, "/device", "VRAM"));
        Assert.Equal(Defaults, ChatSlashCommands.Change(cuda, "/device", "cpu"));
    }

    [Fact]
    public void CanSetAndClearOptionalSettings()
    {
        var configured = ChatSlashCommands.Change(Defaults, "/threads", "8");
        configured = ChatSlashCommands.Change(configured, "/system", "Be brief.");
        configured = ChatSlashCommands.Change(configured, "/seed", "-17");
        Assert.Equal(8, configured.Generation.Threads);
        Assert.Equal(-17, configured.Generation.Seed);
        Assert.Equal("Be brief.", configured.SystemPrompt);
        configured = ChatSlashCommands.Change(configured, "/system", "");
        configured = ChatSlashCommands.Change(configured, "/threads", "default");
        configured = ChatSlashCommands.Change(configured, "/seed", "random");
        Assert.Equal(Defaults, configured);
    }

    [Theory]
    [InlineData("/device", "")]
    [InlineData("/device", "vulkan")]
    [InlineData("/model", "\"\"")]
    [InlineData("/context", "0")]
    [InlineData("/threads", "-1")]
    [InlineData("/temperature", "NaN")]
    [InlineData("/temperature", "0,5")]
    [InlineData("/seed", "abc")]
    [InlineData("/max-tokens", "0")]
    [InlineData("/unknown", "")]
    public void RejectsInvalidCommands(string command, string value)
    {
        Assert.Throws<ArgumentException>(() => ChatSlashCommands.Change(Defaults, command, value));
    }
}
