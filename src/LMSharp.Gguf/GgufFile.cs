using System.Globalization;

namespace LMSharp.Gguf;

public sealed record GgufFile(
    uint Version,
    ulong TensorCount,
    IReadOnlyDictionary<string, GgufMetadataValue> Metadata)
{
    public IReadOnlyList<GgufTensorDescriptor> Tensors { get; init; } =
        Array.Empty<GgufTensorDescriptor>();

    public string? Architecture => Metadata.TryGetValue("general.architecture", out var value)
        ? value.Value as string
        : null;

    public T RequireMetadata<T>(string name) =>
        Metadata.TryGetValue(name, out var entry) && entry.Value is T value
            ? value
            : throw new GgufFormatException($"GGUF metadata '{name}' is missing or has an unexpected type.");
}

public sealed record GgufMetadataValue(
    GgufValueType Type,
    object Value,
    GgufValueType? ElementType = null)
{
    public string ToDisplayString(int previewCount = 3)
    {
        if (Type != GgufValueType.Array)
        {
            return FormatScalar(Value);
        }

        var elements = (IReadOnlyList<object>)Value;
        var preview = string.Join(", ", elements.Take(previewCount).Select(FormatScalar));
        var suffix = elements.Count > previewCount ? ", ..." : string.Empty;
        return $"[{ElementType}; {elements.Count}] {preview}{suffix}";
    }

    private static string FormatScalar(object value) => value switch
    {
        bool boolean => boolean ? "true" : "false",
        string text => text,
        IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
        _ => value.ToString() ?? string.Empty,
    };
}

public sealed class GgufFormatException(string message) : Exception(message);
