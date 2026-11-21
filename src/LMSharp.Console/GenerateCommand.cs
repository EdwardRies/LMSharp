using System.Diagnostics;
using LMSharp.Gguf;
using LMSharp.Inference;

namespace LMSharp.Console;

internal static class GenerateCommand
{
    public const string Usage = "generate --model <path> --prompt <text> " + InferenceArguments.CommonOptions;

    public static async Task<int> RunAsync(string[] args)
    {
        using var cancellation = new CancellationTokenSource();
        ConsoleCancelEventHandler cancel = (_, args) =>
        {
            args.Cancel = true;
            cancellation.Cancel();
        };

        System.Console.CancelKeyPress += cancel;
        try
        {
            var (path, prompt, system, settings, device) = Parse(args);
            var start = Stopwatch.GetTimestamp();
            using var model = await Phi4ModelLoader.LoadAsync(path, device, cancellation.Token);
            var loadTime = Stopwatch.GetElapsedTime(start);
            var result = Phi4Generator.Generate(model, prompt, settings, system, cancellation.Token);
            System.Console.WriteLine(result.Text);

            using var process = Process.GetCurrentProcess();
            System.Console.Error.WriteLine($"Device: {model.DeviceName}");
            System.Console.Error.WriteLine($"Token IDs: {string.Join(" ", result.TokenIds)}");
            System.Console.Error.WriteLine(
                $"Stop: {result.StopReason}. Prompt tokens: {result.PromptTokens}. Output tokens: {result.TokenIds.Count}.");
            System.Console.Error.WriteLine(FormattableString.Invariant(
                $"Model load: {loadTime.TotalSeconds:F3} s. First token after load: {result.TimeToFirstToken.TotalSeconds:F3} s. Peak working set: {process.PeakWorkingSet64 / 1048576.0:F0} MiB."));
            if (result.TokenIds.Count > 1)
            {
                var tokensPerSecond = (result.TokenIds.Count - 1) / result.SubsequentDecodeTime.TotalSeconds;
                System.Console.Error.WriteLine(FormattableString.Invariant(
                    $"Subsequent decoding: {tokensPerSecond:F2} tokens/s."));
            }

            return 0;
        }
        catch (OperationCanceledException) when (cancellation.IsCancellationRequested)
        {
            System.Console.Error.WriteLine("Generation cancelled.");
            return 130;
        }
        catch (Exception exception) when (
            exception is IOException or UnauthorizedAccessException or GgufFormatException or
                ArgumentException or InvalidOperationException)
        {
            System.Console.Error.WriteLine($"Generation failed: {exception.Message}");
            return 1;
        }
        finally
        {
            System.Console.CancelKeyPress -= cancel;
        }
    }

    internal static (string Path, string Prompt, string? System, GenerationOptions Settings, InferenceDevice Device) Parse(string[] args)
    {
        var values = new InferenceArguments(args, Usage, "--prompt");
        return (values.Required("--model"), values.Required("--prompt"), values.Get("--system"),
            values.GetGenerationOptions(), values.GetDevice());
    }
}
