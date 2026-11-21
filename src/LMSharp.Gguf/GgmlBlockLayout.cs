namespace LMSharp.Gguf;

public enum GgmlStorageType : uint
{
    F32 = 0,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
}

public readonly record struct GgmlBlockLayout(int ElementCount, int ByteCount)
{
    public static bool TryGet(GgmlStorageType type, out GgmlBlockLayout layout)
    {
        layout = type switch
        {
            GgmlStorageType.F32 => new(1, 4),
            GgmlStorageType.Q8_0 => new(32, 34),
            GgmlStorageType.Q4_K => new(256, 144),
            GgmlStorageType.Q5_K => new(256, 176),
            GgmlStorageType.Q6_K => new(256, 210),
            _ => default,
        };
        return layout.ElementCount != 0;
    }

    public static GgmlBlockLayout Get(GgmlStorageType type) => TryGet(type, out var layout)
        ? layout
        : throw new ArgumentOutOfRangeException(nameof(type), type, "Unsupported GGML storage type.");
}
