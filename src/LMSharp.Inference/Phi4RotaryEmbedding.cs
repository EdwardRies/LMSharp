namespace LMSharp.Inference;

internal sealed class Phi4RotaryEmbedding
{
    private readonly float[] inverseFrequencies;
    private readonly float[] cosines;
    private readonly float[] sines;
    private readonly float scale;

    public Phi4RotaryEmbedding(float frequencyBase, float scale, ReadOnlySpan<float> shortFactors)
    {
        inverseFrequencies = new float[shortFactors.Length];
        cosines = new float[shortFactors.Length];
        sines = new float[shortFactors.Length];
        this.scale = scale;
        for (var i = 0; i < inverseFrequencies.Length; i++)
        {
            inverseFrequencies[i] = MathF.Pow(frequencyBase, -(float)i / shortFactors.Length) / shortFactors[i];
        }
    }

    public void SetPosition(int position)
    {
        for (var i = 0; i < inverseFrequencies.Length; i++)
        {
            var (sin, cos) = MathF.SinCos(position * inverseFrequencies[i]);
            cosines[i] = cos * scale;
            sines[i] = sin * scale;
        }
    }

    public void RotateHeads(Span<float> heads, int headSize)
    {
        var half = inverseFrequencies.Length;
        for (var start = 0; start < heads.Length; start += headSize)
        {
            for (var i = 0; i < half; i++)
            {
                var first = heads[start + i];
                var second = heads[start + half + i];
                heads[start + i] = first * cosines[i] - second * sines[i];
                heads[start + half + i] = second * cosines[i] + first * sines[i];
            }
        }
    }
}
