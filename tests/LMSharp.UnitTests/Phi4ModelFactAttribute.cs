using Xunit;

namespace LMSharp.UnitTests;

internal sealed class Phi4ModelFactAttribute : FactAttribute
{
    private const string ModelPathVariable = "LMSHARP_PHI4_GGUF";

    public Phi4ModelFactAttribute()
    {
        if (string.IsNullOrWhiteSpace(ModelPath))
        {
            Skip = $"Set {ModelPathVariable} to the supplied Phi-4 Mini GGUF path.";
        }
    }

    public static string ModelPath => Environment.GetEnvironmentVariable(ModelPathVariable)!;
}
