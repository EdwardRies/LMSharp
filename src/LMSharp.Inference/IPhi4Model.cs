using LMSharp.Tokenization;

namespace LMSharp.Inference;

public enum InferenceDevice
{
    Cpu,
    Cuda,
}

public interface IPhi4Model : IDisposable
{
    Phi4Tokenizer Tokenizer { get; }
    int MaximumContextLength { get; }
    string DeviceName { get; }

    /// <summary>Creates a sequential token session. Dispose the session before disposing its model.</summary>
    IPhi4Session CreateSession(int contextLength, int? threads = null);
}

public interface IPhi4Session : IDisposable
{
    int Capacity { get; }
    int Position { get; }
    ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, CancellationToken cancellationToken = default);
    ReadOnlySpan<float> Advance(int token, CancellationToken cancellationToken = default);
    void Rewind(int position);
}

public static class Phi4ModelLoader
{
    public static async Task<IPhi4Model> LoadAsync(
        string path, InferenceDevice device, CancellationToken cancellationToken = default) => device switch
    {
        InferenceDevice.Cpu => await Phi4Model.LoadAsync(path, cancellationToken),
        InferenceDevice.Cuda => await Cuda.CudaPhi4Model.LoadAsync(path, cancellationToken),
        _ => throw new ArgumentOutOfRangeException(nameof(device)),
    };
}
