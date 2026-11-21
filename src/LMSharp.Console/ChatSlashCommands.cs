using LMSharp.Inference;
using static LMSharp.Console.InferenceArguments;

namespace LMSharp.Console;

internal static class ChatSlashCommands
{
    public const string Help = """
        Chat commands
          /help                      Show this menu
          /settings                  Show active settings and context usage
          /device cpu|cuda|toggle     Switch between RAM and VRAM and start a fresh conversation
          /model <path>               Load a supported GGUF and start a fresh conversation
          /context <tokens>           Resize the context and start a fresh conversation
          /system [text]              Set a system prompt and start a fresh conversation. Omit text to clear it
          /threads <n>|default        Set CPU workers. Starts a fresh conversation in CPU mode
          /max-tokens <n>              Set the reply token limit
          /temperature <n>             Set sampling temperature. Use 0 for greedy replies
          /seed <n>|random             Set or clear the sampling seed
          /reset                      Clear the conversation
          /exit                       Close chat
        """;

    public static (string Name, string Value) Split(string command)
    {
        var end = 0;
        while (end < command.Length && !char.IsWhiteSpace(command[end]))
        {
            end++;
        }

        return (command[..end].ToLowerInvariant(), command[end..].Trim());
    }

    public static ChatSettings Change(ChatSettings current, string command, string value) => command switch
    {
        "/device" => current with { Device = DeviceFromValue(value, current.Device) },
        "/model" => current with { ModelPath = ModelPathFromValue(value) },
        "/context" => current with { ContextLength = ParseInteger(value, command, positive: true) },
        "/system" => current with { SystemPrompt = value.Length == 0 ? null : value },
        "/threads" => current with
        {
            Generation = current.Generation with
            {
                Threads = value.Equals("default", StringComparison.OrdinalIgnoreCase)
                    ? null
                    : ParseInteger(value, command, positive: true),
            },
        },
        "/max-tokens" => current with
        {
            Generation = current.Generation with { MaxTokens = ParseInteger(value, command, positive: true) },
        },
        "/temperature" => current with
        {
            Generation = current.Generation with { Temperature = ParseTemperature(value, command) },
        },
        "/seed" => current with
        {
            Generation = current.Generation with
            {
                Seed = value.Equals("random", StringComparison.OrdinalIgnoreCase)
                    ? null
                    : ParseInteger(value, command, positive: false),
            },
        },
        _ => throw new ArgumentException($"Unknown command '{command}'. Type /help for commands."),
    };

    private static InferenceDevice DeviceFromValue(string value, InferenceDevice current) =>
        value.Equals("toggle", StringComparison.OrdinalIgnoreCase)
            ? (current == InferenceDevice.Cpu ? InferenceDevice.Cuda : InferenceDevice.Cpu)
            : value.Equals("vram", StringComparison.OrdinalIgnoreCase)
                ? InferenceDevice.Cuda
                : ParseDevice(value, "/device");

    private static string ModelPathFromValue(string value)
    {
        if (value.Length >= 2 && value[0] is '"' or '\'' && value[0] == value[^1])
        {
            value = value[1..^1];
        }

        return !string.IsNullOrWhiteSpace(value)
            ? value
            : throw new ArgumentException("Use /model <path> to open a supported GGUF file.");
    }
}
