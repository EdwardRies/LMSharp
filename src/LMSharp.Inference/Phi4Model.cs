using LMSharp.Gguf;
using LMSharp.Tensors;
using LMSharp.Tokenization;

namespace LMSharp.Inference;

/// <summary>The packed Phi-4 Mini weights and tokenizer shared by independent inference sessions.</summary>
public sealed class Phi4Model : IPhi4Model
{
    private Phi4Model(
        Phi4Tokenizer tokenizer, ModelShape shape, PackedMatrix embeddings,
        float[] outputNorm, float[] ropeFactors, LayerWeights[] layers)
    {
        Tokenizer = tokenizer;
        Shape = shape;
        Embeddings = embeddings;
        OutputNorm = outputNorm;
        RopeFactors = ropeFactors;
        Layers = layers;
    }

    public Phi4Tokenizer Tokenizer { get; }
    public int MaximumContextLength => Shape.ShortContextLength;
    public string DeviceName => "CPU (RAM)";
    internal ModelShape Shape { get; }
    internal PackedMatrix Embeddings { get; }
    internal float[] OutputNorm { get; }
    internal float[] RopeFactors { get; }
    internal LayerWeights[] Layers { get; }

    public static async Task<Phi4Model> LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        using var file = await new GgufReader().OpenTensorFileAsync(path, cancellationToken);
        var layout = Phi4WeightLayout.Read(file);
        var outputNorm = await ReadVectorAsync(layout.OutputNorm);
        var ropeFactors = await layout.ReadRopeFactorsAsync(file, cancellationToken);
        var embeddings = await ReadMatrixAsync(layout.Embeddings);
        var layers = new LayerWeights[layout.Layers.Length];
        for (var layer = 0; layer < layers.Length; layer++)
        {
            var tensors = layout.Layers[layer];
            layers[layer] = new LayerWeights(
                await ReadVectorAsync(tensors.AttentionNorm), await ReadMatrixAsync(tensors.Qkv),
                await ReadMatrixAsync(tensors.AttentionOutput), await ReadVectorAsync(tensors.FeedForwardNorm),
                await ReadMatrixAsync(tensors.GateUp), await ReadMatrixAsync(tensors.Down));
        }

        return new Phi4Model(layout.Tokenizer, layout.Shape, embeddings, outputNorm, ropeFactors, layers);

        Task<PackedMatrix> ReadMatrixAsync(GgufTensorRange tensor) =>
            PackedMatrix.LoadAsync(file, tensor.Descriptor.Name, cancellationToken);

        Task<float[]> ReadVectorAsync(GgufTensorRange tensor) =>
            Phi4WeightLayout.ReadVectorAsync(file, tensor, cancellationToken);
    }

    public Phi4Session CreateSession(int contextLength = 512, int? threads = null) =>
        new(this, contextLength, threads ?? Math.Min(12, Environment.ProcessorCount));

    IPhi4Session IPhi4Model.CreateSession(int contextLength, int? threads) => CreateSession(contextLength, threads);

    public void Dispose() { }

    internal sealed record LayerWeights(
        float[] AttentionNorm, PackedMatrix Qkv, PackedMatrix AttentionOutput,
        float[] FeedForwardNorm, PackedMatrix GateUp, PackedMatrix Down);

    internal sealed record ModelShape(
        int EmbeddingSize, int FeedForwardSize, int LayerCount, int Heads, int KvHeads,
        int RotarySize, int ShortContextLength, float RmsEpsilon, float FrequencyBase, float RopeScale)
    {
        public int HeadSize => EmbeddingSize / Heads;
        public int KvSize => HeadSize * KvHeads;
        public int QkvSize => EmbeddingSize + 2 * KvSize;

        public static ModelShape Read(GgufFile info)
        {
            if (info.Architecture != "phi3" || info.RequireMetadata<string>("general.name") != "phi-4-mini" ||
                info.RequireMetadata<string>("phi3.rope.scaling.type") != "longrope" ||
                info.RequireMetadata<uint>("general.quantization_version") != 2)
            {
                throw new GgufFormatException("Expected a Phi-4 Mini GGUF with LongRoPE and GGML quantization version 2.");
            }

            return new ModelShape(
                ReadDimension("embedding_length", 3072),
                ReadDimension("feed_forward_length", 8192),
                ReadDimension("block_count", 32),
                ReadDimension("attention.head_count", 24),
                ReadDimension("attention.head_count_kv", 8),
                ReadDimension("rope.dimension_count", 96),
                ReadDimension("rope.scaling.original_context_length", 4096),
                ReadPositive("attention.layer_norm_rms_epsilon"),
                ReadPositive("rope.freq_base"),
                ReadPositive("rope.scaling.attn_factor"));

            int ReadDimension(string key, uint expected)
            {
                if (info.RequireMetadata<uint>("phi3." + key) != expected)
                {
                    throw new GgufFormatException($"Expected phi3.{key} = {expected} for Phi-4 Mini.");
                }

                return (int)expected;
            }

            float ReadPositive(string key)
            {
                var value = info.RequireMetadata<float>("phi3." + key);
                return float.IsFinite(value) && value > 0
                    ? value
                    : throw new GgufFormatException($"Expected a positive finite value for phi3.{key}.");
            }
        }
    }
}
