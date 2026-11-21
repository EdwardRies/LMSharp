using LMSharp.Gguf;
using LMSharp.Tensors;
using LMSharp.Tokenization;

namespace LMSharp.Inference;

internal sealed record Phi4WeightLayout(
    Phi4Model.ModelShape Shape, Phi4Tokenizer Tokenizer, GgufTensorRange Embeddings,
    GgufTensorRange OutputNorm, GgufTensorRange RopeFactors, Phi4WeightLayout.Layer[] Layers)
{
    internal sealed record Layer(
        GgufTensorRange AttentionNorm, GgufTensorRange Qkv, GgufTensorRange AttentionOutput,
        GgufTensorRange FeedForwardNorm, GgufTensorRange GateUp, GgufTensorRange Down);

    public IEnumerable<GgufTensorRange> ModelTensors()
    {
        yield return Embeddings;
        yield return OutputNorm;
        foreach (var layer in Layers)
        {
            yield return layer.AttentionNorm;
            yield return layer.Qkv;
            yield return layer.AttentionOutput;
            yield return layer.FeedForwardNorm;
            yield return layer.GateUp;
            yield return layer.Down;
        }
    }

    public static Phi4WeightLayout Read(GgufTensorFile file)
    {
        var shape = Phi4Model.ModelShape.Read(file.Info);
        var tokenizer = new Phi4Tokenizer(file.Info);
        if (file.Tensors.ContainsKey("output.weight"))
        {
            throw new GgufFormatException("This Phi-4 Mini profile requires tied input and output embeddings.");
        }

        var layers = new Layer[shape.LayerCount];
        for (var layer = 0; layer < layers.Length; layer++)
        {
            var prefix = $"blk.{layer}.";
            layers[layer] = new Layer(
                Vector(prefix + "attn_norm.weight", shape.EmbeddingSize),
                Matrix(prefix + "attn_qkv.weight", shape.EmbeddingSize, shape.QkvSize),
                Matrix(prefix + "attn_output.weight", shape.EmbeddingSize, shape.EmbeddingSize),
                Vector(prefix + "ffn_norm.weight", shape.EmbeddingSize),
                Matrix(prefix + "ffn_up.weight", shape.EmbeddingSize, 2 * shape.FeedForwardSize),
                Matrix(prefix + "ffn_down.weight", shape.FeedForwardSize, shape.EmbeddingSize));
        }

        return new Phi4WeightLayout(shape, tokenizer,
            Matrix("token_embd.weight", shape.EmbeddingSize, tokenizer.VocabularySize),
            Vector("output_norm.weight", shape.EmbeddingSize),
            Vector("rope_factors_short.weight", shape.RotarySize / 2), layers);

        GgufTensorRange Matrix(string name, int columns, int rows) =>
            RequireShape(name, (ulong)columns, (ulong)rows);

        GgufTensorRange Vector(string name, int length)
        {
            var tensor = RequireShape(name, (ulong)length);
            if (tensor.Descriptor.GgmlType != (uint)GgmlStorageType.F32)
            {
                throw new GgufFormatException($"Phi-4 Mini tensor '{name}' must contain F32 values.");
            }

            return tensor;
        }

        GgufTensorRange RequireShape(string name, params ulong[] dimensions)
        {
            if (!file.Tensors.TryGetValue(name, out var tensor) ||
                !tensor.Descriptor.Dimensions.SequenceEqual(dimensions))
            {
                throw new GgufFormatException(
                    $"Expected Phi-4 Mini tensor '{name}' with dimensions {string.Join(" x ", dimensions)}.");
            }

            return tensor;
        }
    }

    public static async Task<float[]> ReadVectorAsync(
        GgufTensorFile file, GgufTensorRange tensor, CancellationToken cancellationToken)
    {
        var bytes = new byte[(int)tensor.ByteLength];
        await file.ReadTensorAsync(tensor.Descriptor.Name, 0, bytes, cancellationToken);
        var values = new float[(int)tensor.Descriptor.Dimensions[0]];
        GgmlDequantizer.Decode(GgmlStorageType.F32, bytes, values);
        if (values.Any(value => !float.IsFinite(value)))
        {
            throw new GgufFormatException($"Phi-4 Mini tensor '{tensor.Descriptor.Name}' contains nonfinite values.");
        }

        return values;
    }

    public async Task<float[]> ReadRopeFactorsAsync(GgufTensorFile file, CancellationToken cancellationToken)
    {
        var factors = await ReadVectorAsync(file, RopeFactors, cancellationToken);
        if (factors.Any(value => value <= 0))
        {
            throw new GgufFormatException("Phi-4 Mini short RoPE factors must be positive.");
        }

        return factors;
    }
}
