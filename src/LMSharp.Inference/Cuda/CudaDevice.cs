using System.Runtime.InteropServices;
using static LMSharp.Inference.Cuda.CudaNative;

namespace LMSharp.Inference.Cuda;

internal sealed class CudaDevice : IDisposable
{
    private readonly int device;
    private nint context;

    public unsafe CudaDevice()
    {
        if (!OperatingSystem.IsWindows() || RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            throw new InvalidOperationException("CUDA mode requires Windows x64.");
        }

        try
        {
            Check(cuInit(0));
            Check(cuDeviceGet(out device, 0));
            var name = stackalloc byte[256];
            Check(cuDeviceGetName(name, 256, device));
            Name = Marshal.PtrToStringUTF8((nint)name)!;
            Check(cuDeviceGetAttribute(out var major, 75, device));
            Check(cuDeviceGetAttribute(out var minor, 76, device));
            ArchitectureTarget = $"sm_{major}{minor}";
            Check(cuDevicePrimaryCtxRetain(out context, device));
        }
        catch (DllNotFoundException exception)
        {
            throw new InvalidOperationException("CUDA requires an NVIDIA driver. Select --device cpu for system RAM.", exception);
        }
    }

    public string Name { get; }
    public string ArchitectureTarget { get; }

    public ContextScope Bind()
    {
        ObjectDisposedException.ThrowIf(context == 0, this);
        Check(cuCtxPushCurrent_v2(context));
        return new ContextScope();
    }

    public unsafe nint LoadModule(byte[] image)
    {
        using var scope = Bind();
        fixed (byte* data = image)
        {
            Check(cuModuleLoadData(out var module, data));
            return module;
        }
    }

    public DeviceAllocation Allocate(long bytes)
    {
        using var scope = Bind();
        Check(cuMemGetInfo_v2(out var free, out _));
        if ((nuint)bytes > free)
        {
            throw new InvalidOperationException(
                $"CUDA needs {bytes / 1048576} MiB for this allocation and has {free / 1048576} MiB free.");
        }

        Check(cuMemAlloc_v2(out var address, (nuint)bytes));
        return new DeviceAllocation(this, address);
    }

    public unsafe void Upload<T>(ulong destination, ReadOnlySpan<T> data) where T : unmanaged
    {
        using var scope = Bind();
        fixed (T* source = data)
        {
            Check(cuMemcpyHtoD_v2(destination, source, (nuint)data.Length * (nuint)sizeof(T)));
        }
    }

    public static unsafe void Download<T>(Span<T> destination, ulong source) where T : unmanaged
    {
        fixed (T* target = destination)
        {
            Check(cuMemcpyDtoH_v2(target, source, (nuint)destination.Length * (nuint)sizeof(T)));
        }
    }

    public void Dispose()
    {
        if (context != 0)
        {
            context = 0;
            Check(cuDevicePrimaryCtxRelease_v2(device));
        }
    }

    public readonly ref struct ContextScope
    {
        public void Dispose() => Check(cuCtxPopCurrent_v2(out _));
    }

    internal sealed class DeviceAllocation(CudaDevice device, ulong address) : IDisposable
    {
        public ulong Address { get; private set; } = address;

        public void Dispose()
        {
            if (Address != 0)
            {
                using var scope = device.Bind();
                Check(cuMemFree_v2(Address));
                Address = 0;
            }
        }
    }
}
