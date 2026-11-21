using System.Globalization;
using LMSharp.Inference;

namespace LMSharp.Console;

internal sealed class InferenceArguments
{
    public const string CommonOptions = "[--device cpu|cuda] [--system <text>] [--max-tokens <n>] [--threads <n>] " +
        "[--temperature <value>] [--seed <n>]";

    private readonly Dictionary<string, string> values = new(StringComparer.Ordinal);
    private readonly string usage;

    public InferenceArguments(string[] args, string usage, string modeOption)
    {
        this.usage = usage;
        for (var index = 1; index < args.Length; index += 2)
        {
            var flag = args[index];
            if ((flag != modeOption && flag is not ("--model" or "--device" or "--system" or "--max-tokens" or "--threads" or
                "--temperature" or "--seed")) || index + 1 >= args.Length || !values.TryAdd(flag, args[index + 1]))
            {
                throw new ArgumentException($"Expected one value for each supported option. Usage: {usage}");
            }
        }
    }

    public string? Get(string flag) => values.GetValueOrDefault(flag);

    public string Required(string flag) => Get(flag) is { } value && !string.IsNullOrWhiteSpace(value)
        ? value
        : throw new ArgumentException($"{flag} is required. Usage: {usage}");

    public InferenceDevice GetDevice() => ParseDevice(Get("--device"), "--device");

    public static InferenceDevice ParseDevice(string? value, string option) => value?.ToLowerInvariant() switch
    {
        null or "cpu" => InferenceDevice.Cpu,
        "cuda" => InferenceDevice.Cuda,
        _ => throw new ArgumentException($"{option} must be cpu or cuda."),
    };

    public GenerationOptions GetGenerationOptions()
    {
        var maxTokens = GetInteger("--max-tokens", positive: true) ?? new GenerationOptions().MaxTokens;
        var threads = GetInteger("--threads", positive: true);
        var seed = GetInteger("--seed", positive: false);
        var temperature = Get("--temperature") is { } rawTemperature
            ? ParseTemperature(rawTemperature, "--temperature")
            : 0;
        return new GenerationOptions(maxTokens, temperature, seed, threads);
    }

    public static float ParseTemperature(string value, string option)
    {
        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var temperature) ||
            !float.IsFinite(temperature) || temperature < 0)
        {
            throw new ArgumentException($"{option} must be a finite nonnegative number.");
        }

        return temperature;
    }

    public int? GetInteger(string flag, bool positive) =>
        Get(flag) is { } value ? ParseInteger(value, flag, positive) : null;

    public static int ParseInteger(string value, string flag, bool positive)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) ||
            (positive && parsed <= 0))
        {
            throw new ArgumentException($"{flag} must be {(positive ? "a positive integer" : "an integer")}.");
        }

        return parsed;
    }
}
