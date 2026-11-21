using System.Diagnostics;
using LMSharp.Tokenization;

namespace LMSharp.Inference;

public sealed class ChatContextFullException(int contextLength)
    : InvalidOperationException($"There is insufficient room for another turn in the {contextLength}-token context.");

/// <summary>Retains the model's token context across a sequence of chat turns.</summary>
public sealed class Phi4ChatSession : IDisposable
{
    public const int DefaultContextLength = 1024;

    private readonly Phi4Tokenizer tokenizer;
    private readonly IPhi4Session session;
    private GenerationOptions settings;
    private readonly string? systemPrompt;
    private TokenSampler sampler;
    private int[] pendingAssistantTail = [];

    public Phi4ChatSession(
        IPhi4Model model, int contextLength = DefaultContextLength,
        GenerationOptions? settings = null, string? systemPrompt = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentOutOfRangeException.ThrowIfLessThan(contextLength, 6);
        this.settings = settings ?? new GenerationOptions();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(this.settings.MaxTokens);
        sampler = new TokenSampler(this.settings.Temperature, this.settings.Seed);
        session = model.CreateSession(contextLength, this.settings.Threads);
        tokenizer = model.Tokenizer;
        this.systemPrompt = systemPrompt;
    }

    public int ContextLength => session.Capacity;
    public int ContextTokensUsed => session.Position + pendingAssistantTail.Length;

    public void UpdateReplyOptions(int maxTokens, float temperature, int? seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxTokens);
        if (temperature != settings.Temperature || seed != settings.Seed)
        {
            sampler = new TokenSampler(temperature, seed);
        }

        settings = settings with { MaxTokens = maxTokens, Temperature = temperature, Seed = seed };
    }

    public GenerationResult Reply(string question, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(question);
        cancellationToken.ThrowIfCancellationRequested();
        var start = Stopwatch.GetTimestamp();
        var turn = tokenizer.Encode(Phi4Tokenizer.FormatPrompt(question, session.Position == 0 ? systemPrompt : null));
        var available = ContextLength - ContextTokensUsed - turn.Count;
        // Reserve a closing turn marker even if the model uses the full reply budget.
        if (available < 2)
        {
            throw new ChatContextFullException(ContextLength);
        }

        var prompt = pendingAssistantTail.Length == 0 ? turn : [.. pendingAssistantTail, .. turn];
        var previousPosition = session.Position;
        try
        {
            var result = Phi4Generator.GenerateTurn(
                tokenizer, session, prompt, Math.Min(settings.MaxTokens, available - 1), sampler, start,
                cancellationToken);
            // GenerateTurn leaves the last sampled token out of the cache. Carry text
            // forward as-is and use the template's end-of-turn token for the boundary.
            pendingAssistantTail = result.StopReason == GenerationStopReason.EndOfSequence
                ? [tokenizer.EndOfTurnId]
                : [result.TokenIds[^1], tokenizer.EndOfTurnId];
            return result;
        }
        catch
        {
            session.Rewind(previousPosition);
            throw;
        }
    }

    public void Reset()
    {
        session.Rewind(0);
        pendingAssistantTail = [];
        sampler = new TokenSampler(settings.Temperature, settings.Seed);
    }

    public void Dispose() => session.Dispose();
}
