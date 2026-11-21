using LMSharp.Inference;

namespace LMSharp.Console;

internal sealed record ChatSettings(
    string ModelPath, int ContextLength, string? SystemPrompt, GenerationOptions Generation, InferenceDevice Device);

internal enum ChatChange
{
    None,
    ReplyOptions,
    NewContext,
    NewModel,
}

/// <summary>Owns the current model and its conversation while the terminal stays open.</summary>
internal sealed class ChatRuntime : IDisposable
{
    private IPhi4Model model;

    private ChatRuntime(ChatSettings settings, IPhi4Model model, Phi4ChatSession conversation)
    {
        Settings = settings;
        this.model = model;
        Conversation = conversation;
    }

    public ChatSettings Settings { get; private set; }
    public Phi4ChatSession Conversation { get; private set; }
    public string DeviceName => model.DeviceName;

    public static async Task<ChatRuntime> OpenAsync(ChatSettings settings)
    {
        settings = settings with { ModelPath = Path.GetFullPath(settings.ModelPath) };
        var (model, conversation) = await OpenModelAsync(settings);
        return new ChatRuntime(settings, model, conversation);
    }

    public async Task<ChatChange> UpdateAsync(ChatSettings next)
    {
        var path = Path.GetFullPath(next.ModelPath);
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        var samePath = string.Equals(path, Settings.ModelPath, comparison);
        next = next with { ModelPath = samePath ? Settings.ModelPath : path };
        if (next == Settings)
        {
            return ChatChange.None;
        }

        if (!samePath || next.Device != Settings.Device)
        {
            // Open the replacement first so a missing file or unavailable GPU
            // leaves the current conversation usable.
            var (nextModel, nextConversation) = await OpenModelAsync(next);
            var previousModel = model;
            var previousConversation = Conversation;
            model = nextModel;
            Conversation = nextConversation;
            Settings = next;
            Release(previousModel, previousConversation);
            return ChatChange.NewModel;
        }

        if (next.ContextLength != Settings.ContextLength || next.SystemPrompt != Settings.SystemPrompt ||
            (next.Device == InferenceDevice.Cpu && next.Generation.Threads != Settings.Generation.Threads))
        {
            var nextConversation = CreateConversation(model, next);
            var previousConversation = Conversation;
            Conversation = nextConversation;
            Settings = next;
            previousConversation.Dispose();
            return ChatChange.NewContext;
        }

        Conversation.UpdateReplyOptions(next.Generation.MaxTokens, next.Generation.Temperature, next.Generation.Seed);
        Settings = next;
        return ChatChange.ReplyOptions;
    }

    public void Dispose() => Release(model, Conversation);

    private static async Task<(IPhi4Model, Phi4ChatSession)> OpenModelAsync(ChatSettings settings)
    {
        var model = await Phi4ModelLoader.LoadAsync(settings.ModelPath, settings.Device);
        try
        {
            return (model, CreateConversation(model, settings));
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    private static Phi4ChatSession CreateConversation(IPhi4Model model, ChatSettings settings) =>
        new(model, settings.ContextLength, settings.Generation, settings.SystemPrompt);

    private static void Release(IPhi4Model model, Phi4ChatSession conversation)
    {
        try
        {
            conversation.Dispose();
        }
        finally
        {
            model.Dispose();
        }
    }
}
