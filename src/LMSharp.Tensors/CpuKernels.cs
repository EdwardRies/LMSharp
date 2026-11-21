using System.Numerics;

namespace LMSharp.Tensors;

public static class CpuKernels
{
    public static float Dot(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Dot operands must have the same length.");
        }

        var i = 0;
        var width = Vector<float>.Count;
        var vectorSum = Vector<float>.Zero;
        for (; i <= left.Length - width; i += width)
        {
            vectorSum += new Vector<float>(left[i..]) * new Vector<float>(right[i..]);
        }

        var sum = Vector.Sum(vectorSum);
        for (; i < left.Length; i++)
        {
            sum += left[i] * right[i];
        }

        return sum;
    }

    public static void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, float epsilon, Span<float> output)
    {
        if (input.IsEmpty || input.Length != weights.Length || input.Length != output.Length ||
            !float.IsFinite(epsilon) || epsilon <= 0)
        {
            throw new ArgumentException("RMS normalization needs equal, nonempty vectors and positive finite epsilon.");
        }

        double sumSquares = 0;
        foreach (var value in input)
        {
            sumSquares += (double)value * value;
        }

        var scale = 1 / MathF.Sqrt((float)(sumSquares / input.Length) + epsilon);
        for (var i = 0; i < input.Length; i++)
        {
            output[i] = input[i] * scale * weights[i];
        }
    }

    public static void SwiGlu(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> output)
    {
        if (gate.Length != up.Length || gate.Length != output.Length)
        {
            throw new ArgumentException("SwiGLU requires equal-length operands.");
        }

        for (var i = 0; i < gate.Length; i++)
        {
            var activation = gate[i] / (1 + MathF.Exp(-gate[i]));
            output[i] = activation * up[i];
        }
    }

    public static void SoftmaxInPlace(Span<float> scores)
    {
        if (scores.IsEmpty)
        {
            throw new ArgumentException("Softmax needs at least one score.", nameof(scores));
        }

        var max = float.NegativeInfinity;
        foreach (var score in scores)
        {
            max = MathF.Max(max, score);
        }

        double sum = 0;
        for (var i = 0; i < scores.Length; i++)
        {
            scores[i] = MathF.Exp(scores[i] - max);
            sum += scores[i];
        }

        var inverse = (float)(1 / sum);
        for (var i = 0; i < scores.Length; i++)
        {
            scores[i] *= inverse;
        }
    }
}
