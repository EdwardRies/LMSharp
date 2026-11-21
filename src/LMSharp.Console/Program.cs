using LMSharp.Gguf;
using LMSharp.Tokenization;

namespace LMSharp.Console;

public static class Program
{
    public static string Run(string[] args) => args.Contains("--version", StringComparer.Ordinal)
        ? "1.0.0"
        : "Usage: LMSharp.Console inspect --model <path> [--tensor <name>] | " +
            "tokenize --model <path> --text <text> | " + GenerateCommand.Usage + " | " + ChatCommand.Usage + " | --version";

    public static async Task<int> Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "chat")
        {
            return await ChatCommand.RunAsync(args);
        }

        if (args.Length > 0 && args[0] == "generate")
        {
            return await GenerateCommand.RunAsync(args);
        }

        if (args.Length == 5 && args[0] == "tokenize" && args[1] == "--model" && args[3] == "--text")
        {
            try
            {
                var model = await new GgufReader().ReadAsync(args[2]);
                var tokenizer = new Phi4Tokenizer(model);
                System.Console.WriteLine(string.Join(" ", tokenizer.Encode(args[4])));
                return 0;
            }
            catch (Exception exception) when (
                exception is IOException or UnauthorizedAccessException or GgufFormatException)
            {
                System.Console.Error.WriteLine($"Tokenization failed: {exception.Message}");
                return 1;
            }
        }

        if ((args.Length == 3 || (args.Length == 5 && args[3] == "--tensor")) &&
            string.Equals(args[0], "inspect", StringComparison.Ordinal) &&
            string.Equals(args[1], "--model", StringComparison.Ordinal))
        {
            try
            {
                var reader = new GgufReader();
                if (args.Length == 5)
                {
                    return await InspectTensorAsync(reader, args[2], args[4]);
                }

                var model = await reader.ReadAsync(args[2]);
                System.Console.WriteLine($"Architecture: {model.Architecture ?? "(missing)"}");
                System.Console.WriteLine($"Tensor count: {model.TensorCount}");
                System.Console.WriteLine($"Metadata ({model.Metadata.Count} entries):");
                foreach (var entry in model.Metadata.OrderBy(item => item.Key, StringComparer.Ordinal))
                {
                    System.Console.WriteLine($"  {entry.Key} = {entry.Value.ToDisplayString()}");
                }

                return 0;
            }
            catch (Exception exception) when (
                exception is IOException or UnauthorizedAccessException or GgufFormatException)
            {
                System.Console.Error.WriteLine($"GGUF inspection failed: {exception.Message}");
                return 1;
            }
        }

        System.Console.WriteLine(Run(args));
        return args.Contains("--version", StringComparer.Ordinal) ? 0 : 1;
    }

    private static async Task<int> InspectTensorAsync(GgufReader reader, string path, string name)
    {
        using var file = await reader.OpenTensorFileAsync(path);
        if (!file.Tensors.TryGetValue(name, out var tensor))
        {
            System.Console.Error.WriteLine($"Tensor '{name}' was not found.");
            return 1;
        }

        var prefix = new byte[(int)Math.Min(32, tensor.ByteLength)];
        await file.ReadTensorAsync(name, 0, prefix);
        System.Console.WriteLine($"Tensor: {name}");
        System.Console.WriteLine($"Dimensions: {string.Join(" x ", tensor.Descriptor.Dimensions)}");
        System.Console.WriteLine($"GGML type ID: {tensor.Descriptor.GgmlType}");
        System.Console.WriteLine($"File offset: 0x{tensor.FileOffset:X}");
        System.Console.WriteLine($"Byte length: {tensor.ByteLength}");
        System.Console.WriteLine($"First {prefix.Length} bytes: {Convert.ToHexString(prefix)}");
        return 0;
    }
}
