using LMSharp.Gguf;
using LMSharp.Inference;

namespace LMSharp.Console;

internal static class ChatCommand
{
    public const string Usage = "chat --model <path> [--context <n>] " + InferenceArguments.CommonOptions;
    public static async Task<int> RunAsync(string[] args)
    {
        try
        {
            var settings = Parse(args);
            System.Console.WriteLine("Loading Phi-4 Mini...");
            using var chat = await ChatRuntime.OpenAsync(settings);
            await RunLoopAsync(chat, System.Console.In, System.Console.Out);
            return 0;
        }
        catch (Exception exception) when (IsChatError(exception))
        {
            System.Console.Error.WriteLine($"Chat failed: {exception.Message}");
            return 1;
        }
    }

    internal static ChatSettings Parse(string[] args)
    {
        var values = new InferenceArguments(args, Usage, "--context");
        return new ChatSettings(values.Required("--model"),
            values.GetInteger("--context", positive: true) ?? Phi4ChatSession.DefaultContextLength,
            values.Get("--system"), values.GetGenerationOptions(), values.GetDevice());
    }

    internal static async Task RunLoopAsync(ChatRuntime chat, TextReader input, TextWriter output)
    {
        output.WriteLine($"Device: {chat.DeviceName}");
        output.WriteLine($"LMSharp chat · {chat.Conversation.ContextLength:N0}-token context");
        output.WriteLine("Type a question and press Enter, or type /help for chat commands.");
        while (true)
        {
            output.Write("You> ");
            output.Flush();
            var line = await input.ReadLineAsync();
            if (line is null)
            {
                output.WriteLine();
                return;
            }

            var command = line.Trim();
            if (command.Length == 0)
            {
                continue;
            }

            if (command.StartsWith('/'))
            {
                if (!await HandleCommandAsync(chat, command, output))
                {
                    return;
                }

                continue;
            }

            var conversation = chat.Conversation;
            output.Write("Assistant> ");
            output.Flush();
            try
            {
                var reply = conversation.Reply(line);
                output.WriteLine(reply.Text.Length > 0 ? reply.Text : "(No text in the reply.)");
                if (reply.StopReason == GenerationStopReason.TokenLimit)
                {
                    output.WriteLine("Reply reached its token limit.");
                }

                output.WriteLine($"Context used: {conversation.ContextTokensUsed:N0}/{conversation.ContextLength:N0} tokens.");
            }
            catch (ChatContextFullException exception)
            {
                output.WriteLine();
                output.WriteLine($"{exception.Message} Try a shorter question or type /reset.");
            }

            output.WriteLine();
        }
    }

    private static async Task<bool> HandleCommandAsync(ChatRuntime chat, string line, TextWriter output)
    {
        var (command, value) = ChatSlashCommands.Split(line);
        try
        {
            if ((command is "/exit" or "/reset" or "/help" or "/settings") && value.Length != 0)
            {
                throw new ArgumentException($"Use {command} without arguments.");
            }

            switch (command)
            {
                case "/exit":
                    return false;
                case "/reset":
                    chat.Conversation.Reset();
                    output.WriteLine("Conversation cleared.");
                    break;
                case "/help":
                    output.WriteLine(ChatSlashCommands.Help);
                    break;
                case "/settings":
                    WriteSettings(chat, output);
                    break;
                default:
                    var next = ChatSlashCommands.Change(chat.Settings, command, value);
                    var change = await chat.UpdateAsync(next);
                    output.WriteLine(change switch
                    {
                        ChatChange.None => "Those settings are already active.",
                        ChatChange.ReplyOptions => $"{command} updated. Conversation retained.",
                        ChatChange.NewContext => $"{command} updated. Conversation reset.",
                        ChatChange.NewModel => $"Model loaded on {chat.DeviceName}. Conversation reset.",
                        _ => throw new InvalidOperationException("Unexpected chat setting result."),
                    });
                    break;
            }
        }
        catch (Exception exception) when (IsChatError(exception))
        {
            output.WriteLine($"Command failed: {exception.Message}");
        }

        return true;
    }

    private static void WriteSettings(ChatRuntime chat, TextWriter output)
    {
        var settings = chat.Settings;
        var generation = settings.Generation;
        output.WriteLine($"Model: {settings.ModelPath}");
        output.WriteLine($"Device: {chat.DeviceName}");
        output.WriteLine($"Context: {chat.Conversation.ContextTokensUsed:N0}/{settings.ContextLength:N0} tokens used");
        output.WriteLine($"Max reply tokens: {generation.MaxTokens:N0}");
        output.WriteLine(FormattableString.Invariant($"Temperature: {generation.Temperature}"));
        output.WriteLine($"Seed: {generation.Seed?.ToString() ?? "random"}");
        output.WriteLine($"CPU threads: {generation.Threads?.ToString() ?? "default"}");
        output.WriteLine($"System prompt: {settings.SystemPrompt ?? "(none)"}");
    }

    private static bool IsChatError(Exception exception) =>
        exception is IOException or UnauthorizedAccessException or GgufFormatException or
            ArgumentException or InvalidOperationException;
}
