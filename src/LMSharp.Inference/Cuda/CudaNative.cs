using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace LMSharp.Inference.Cuda;

internal static unsafe class CudaNative
{
    private const string Driver = "nvcuda.dll";

    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuInit(uint flags);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuDeviceGet(out int device, int ordinal);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuDeviceGetName(byte* name, int length, int device);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuDeviceGetAttribute(out int value, int attribute, int device);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuDevicePrimaryCtxRetain(out nint context, int device);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuDevicePrimaryCtxRelease_v2(int device);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuCtxPushCurrent_v2(nint context);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuCtxPopCurrent_v2(out nint context);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuCtxSynchronize();
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuMemGetInfo_v2(out nuint free, out nuint total);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuMemAlloc_v2(out ulong pointer, nuint size);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuMemFree_v2(ulong pointer);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuMemcpyHtoD_v2(ulong destination, void* source, nuint bytes);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuMemcpyDtoH_v2(void* destination, ulong source, nuint bytes);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuModuleLoadData(out nint module, byte* image);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuModuleUnload(nint module);
    [DllImport(Driver, CharSet = CharSet.Ansi, ExactSpelling = true)]
    [DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    public static extern int cuModuleGetFunction(out nint function, nint module, string name);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    private static extern int cuGetErrorName(int result, out nint name);
    [DllImport(Driver), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
    private static extern int cuLaunchKernel(nint function, uint gridX, uint gridY, uint gridZ,
        uint blockX, uint blockY, uint blockZ, uint sharedBytes, nint stream, nint* arguments, nint extra);

    public static void Check(int result, [CallerArgumentExpression(nameof(result))] string? operation = null)
    {
        if (result != 0)
        {
            var name = cuGetErrorName(result, out var nativeName) == 0
                ? Marshal.PtrToStringAnsi(nativeName)
                : null;
            throw new InvalidOperationException($"CUDA {operation} failed ({name ?? "error"}, {result}).");
        }
    }

    public static void Launch(nint function, int blocks, int threads, ReadOnlySpan<ulong> arguments, int sharedBytes = 0)
    {
        var pointers = stackalloc nint[arguments.Length];
        fixed (ulong* values = arguments)
        {
            for (var index = 0; index < arguments.Length; index++)
            {
                pointers[index] = (nint)(values + index);
            }

            Check(cuLaunchKernel(function, (uint)blocks, 1, 1, (uint)threads, 1, 1, (uint)sharedBytes, 0, pointers, 0));
        }
    }
}
