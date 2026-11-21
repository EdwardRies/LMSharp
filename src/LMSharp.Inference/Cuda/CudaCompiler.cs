using System.Runtime.InteropServices;
using System.Text;

namespace LMSharp.Inference.Cuda;

internal static unsafe class CudaCompiler
{
    private const string CompilerOverride = "LMSHARP_NVRTC_DLL";

    public static byte[] Compile(string architecture)
    {
        var path = FindLibrary();
        nint library = 0;
        nint builtins = 0;
        try
        {
            library = NativeLibrary.Load(path);
            var version = (delegate* unmanaged[Cdecl]<int*, int*, int>)NativeLibrary.GetExport(library, "nvrtcVersion");
            int major, minor;
            ExpectSuccess(version(&major, &minor), "query the compiler version");
            // NVRTC resolves this dependency by DLL name. Preload its matching
            // companion from the same installation so PATH stays untouched.
            var companion = Path.Combine(Path.GetDirectoryName(path)!, $"nvrtc-builtins64_{major}{minor}.dll");
            builtins = NativeLibrary.Load(companion);
            return Compile(library, architecture);
        }
        catch (Exception exception) when (
            exception is DllNotFoundException or BadImageFormatException or EntryPointNotFoundException)
        {
            throw new InvalidOperationException(
                $"Could not load NVRTC and its matching built-ins from '{Path.GetDirectoryName(path)}'.", exception);
        }
        finally
        {
            if (library != 0) NativeLibrary.Free(library);
            if (builtins != 0) NativeLibrary.Free(builtins);
        }
    }

    private static byte[] Compile(nint library, string architecture)
    {
        var create = (delegate* unmanaged[Cdecl]<nint*, byte*, byte*, int, nint, nint, int>)
            NativeLibrary.GetExport(library, "nvrtcCreateProgram");
        var compile = (delegate* unmanaged[Cdecl]<nint, int, byte**, int>)
            NativeLibrary.GetExport(library, "nvrtcCompileProgram");
        var destroy = (delegate* unmanaged[Cdecl]<nint*, int>)NativeLibrary.GetExport(library, "nvrtcDestroyProgram");
        using var resource = typeof(CudaCompiler).Assembly.GetManifestResourceStream("LMSharp.Inference.Cuda.Phi4.cu")
            ?? throw new InvalidOperationException("The LMSharp CUDA kernel resource is missing.");
        using var reader = new StreamReader(resource);
        var source = Encoding.UTF8.GetBytes(reader.ReadToEnd() + "\0");
        var target = Encoding.ASCII.GetBytes($"--gpu-architecture={architecture}\0");
        nint program = 0;
        fixed (byte* code = source)
        {
            ExpectSuccess(create(&program, code, null, 0, 0, 0), "create a CUDA program");
        }

        try
        {
            fixed (byte* targetOption = target)
            {
                var status = compile(program, 1, &targetOption);
                if (status != 0)
                {
                    var log = ReadOutput(library, program, "ProgramLog");
                    throw new InvalidOperationException(
                        $"NVRTC compilation for {architecture} failed ({status}). {Encoding.UTF8.GetString(log).TrimEnd('\0')}");
                }
            }

            return ReadOutput(library, program, "CUBIN");
        }
        finally
        {
            destroy(&program);
        }
    }

    private static byte[] ReadOutput(nint library, nint program, string kind)
    {
        var sizeFunction = (delegate* unmanaged[Cdecl]<nint, nuint*, int>)
            NativeLibrary.GetExport(library, $"nvrtcGet{kind}Size");
        var copyFunction = (delegate* unmanaged[Cdecl]<nint, byte*, int>)
            NativeLibrary.GetExport(library, $"nvrtcGet{kind}");
        nuint size = 0;
        ExpectSuccess(sizeFunction(program, &size), $"get {kind} size");
        var result = new byte[checked((int)size)];
        fixed (byte* destination = result)
        {
            ExpectSuccess(copyFunction(program, destination), $"get {kind}");
        }

        return result;
    }

    private static void ExpectSuccess(int status, string operation)
    {
        if (status != 0)
        {
            throw new InvalidOperationException($"NVRTC could not {operation} ({status}).");
        }
    }

    private static string FindLibrary()
    {
        var explicitPath = Environment.GetEnvironmentVariable(CompilerOverride);
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            return File.Exists(explicitPath)
                ? Path.GetFullPath(explicitPath)
                : throw new InvalidOperationException($"{CompilerOverride} must point to an existing NVRTC DLL.");
        }

        var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var localBin = Path.Combine(localAppData, "LMSharp", "cuda-runtime", "nvidia", "cu13", "bin", "x86_64");
        var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        var folders = new List<string> { localBin };
        if (!string.IsNullOrWhiteSpace(cudaPath))
        {
            folders.Add(Path.Combine(cudaPath, "bin", "x64"));
            folders.Add(Path.Combine(cudaPath, "bin"));
        }

        foreach (var folder in folders)
        {
            foreach (var dll in new[] { "nvrtc64_130_0.dll", "nvrtc64_120_0.dll" })
            {
                var path = Path.Combine(folder, dll);
                if (File.Exists(path))
                {
                    return path;
                }
            }
        }

        throw new InvalidOperationException(
            $"NVRTC was not found. Install the compatible NVIDIA compiler runtime or set {CompilerOverride}. See the LMSharp README.");
    }
}
