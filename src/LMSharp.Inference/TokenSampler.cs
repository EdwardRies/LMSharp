namespace LMSharp.Inference;

internal sealed class TokenSampler
{
    private readonly float temperature;
    private readonly Random random;

    public TokenSampler(float temperature, int? seed)
    {
        if (!float.IsFinite(temperature) || temperature < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be finite and nonnegative.");
        }

        this.temperature = temperature;
        random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    public int Choose(ReadOnlySpan<float> logits)
    {
        if (logits.IsEmpty)
        {
            throw new ArgumentException("At least one token score is required.", nameof(logits));
        }

        var best = 0;
        for (var i = 0; i < logits.Length; i++)
        {
            if (!float.IsFinite(logits[i]))
            {
                throw new InvalidOperationException($"The model produced a nonfinite score for token {i}.");
            }

            if (logits[i] > logits[best])
            {
                best = i;
            }
        }

        if (temperature == 0)
        {
            return best;
        }

        var max = logits[best];
        double total = 0;
        foreach (var logit in logits)
        {
            total += Math.Exp(((double)logit - max) / temperature);
        }

        var remaining = random.NextDouble() * total;
        var lastWithWeight = best;
        for (var i = 0; i < logits.Length; i++)
        {
            var weight = Math.Exp(((double)logits[i] - max) / temperature);
            if (weight > 0)
            {
                lastWithWeight = i;
                remaining -= weight;
                if (remaining < 0)
                {
                    return i;
                }
            }
        }

        return lastWithWeight;
    }
}
