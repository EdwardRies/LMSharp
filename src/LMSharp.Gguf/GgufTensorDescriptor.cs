namespace LMSharp.Gguf;

public sealed record GgufTensorDescriptor(
    string Name,
    IReadOnlyList<ulong> Dimensions,
    uint GgmlType,
    ulong Offset)
{
    public int DimensionCount => Dimensions.Count;
}
