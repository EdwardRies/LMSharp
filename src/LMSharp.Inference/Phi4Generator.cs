using System.Diagnostics;
using LMSharp.Tokenization;

namespace LMSharp.Inference;

public sealed record GenerationOptions(
    int MaxTokens = 64, float Temperature = 0, int? Seed = null, int? Threads = null);

public enum GenerationStopReason
{
    EndOfSequence,
    TokenLimit,
}

public sealed record GenerationResult(
    string Text, IReadOnlyList<int> TokenIds, GenerationStopReason StopReason,
    int PromptTokens, TimeSpan TimeToFirstToken, TimeSpan SubsequentDecodeTime);

public static class Phi4Generator
{
    public static GenerationResult Generate(
        IPhi4Model model, string userPrompt, GenerationOptions? settings = null,
        string? systemPrompt = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(userPrompt);
        settings ??= new GenerationOptions();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(settings.MaxTokens);
        var sampler = new TokenSampler(settings.Temperature, settings.Seed);
        cancellationToken.ThrowIfCancellationRequested();

        var start = Stopwatch.GetTimestamp();
        var tokenizer = model.Tokenizer;
        var prompt = tokenizer.Encode(Phi4Tokenizer.FormatPrompt(userPrompt, systemPrompt));
        if (prompt.Count > model.MaximumContextLength - settings.MaxTokens)
        {
            throw new ArgumentException(
                $"The prompt and output limit must fit in the {model.MaximumContextLength}-token context.");
        }

        using var session = model.CreateSession(prompt.Count + settings.MaxTokens, settings.Threads);
        return GenerateTurn(tokenizer, session, prompt, settings.MaxTokens, sampler, start, cancellationToken);
    }

    internal static GenerationResult GenerateTurn(
        Phi4Tokenizer tokenizer, IPhi4Session session, IReadOnlyList<int> prompt, int maxTokens,
        TokenSampler sampler, long start, CancellationToken cancellationToken)
    {
        var ids = new List<int>(maxTokens);
        var nextLogits = session.Prefill(prompt, cancellationToken);
        var next = sampler.Choose(nextLogits);
        var first = Stopwatch.GetTimestamp();
        var reason = GenerationStopReason.TokenLimit;
        for (var index = 0; index < maxTokens; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (index > 0)
            {
                next = sampler.Choose(session.Advance(next, cancellationToken));
            }

            ids.Add(next);
            if (next == tokenizer.EndOfTextId || next == tokenizer.EndOfTurnId)
            {
                reason = GenerationStopReason.EndOfSequence;
                break;
            }
        }

        var finished = Stopwatch.GetTimestamp();
        return new GenerationResult(tokenizer.Decode(ids), ids.AsReadOnly(), reason, prompt.Count,
            Stopwatch.GetElapsedTime(start, first), Stopwatch.GetElapsedTime(first, finished));
    }
}
