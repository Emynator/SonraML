using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Enums;

namespace SonraML.Backend.MLX.Interop;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxDevice
{
    private void* ctx;
    
    [LibraryImport("mlxc", EntryPoint = "mlx_device_new")]
    public static partial MlxDevice New();
    
    [LibraryImport("mlxc", EntryPoint = "mlx_device_new_type")]
    public static partial MlxDevice NewType(MlxDeviceType type, int index);

    [LibraryImport("mlxc", EntryPoint = "mlx_device_free")]
    public static partial int Free(MlxDevice dev);

    [LibraryImport("mlxc", EntryPoint = "mlx_device_set")]
    public static partial int Set(ref MlxDevice dev, MlxDevice src);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_device_tostring")]
    public static partial int ToString(out MlxString str, MlxDevice dev);

    [LibraryImport("mlxc", EntryPoint = "mlx_device_equal")]
    [return: MarshalAs(UnmanagedType.U1)]
    public static partial bool Equal(MlxDevice lhs, MlxDevice rhs);

    [LibraryImport("mlxc", EntryPoint = "mlx_device_get_index")]
    public static partial int GetIndex(out int index, MlxDevice dev);

    [LibraryImport("mlxc", EntryPoint = "mlx_device_get_type")]
    public static partial int GetType(out MlxDeviceType type, MlxDevice dev);

    [LibraryImport("mlxc", EntryPoint = "mlx_get_default_device")]
    public static partial int GetDefaultDevice(ref readonly MlxDevice dev);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_set_default_device")]
    public static partial int SetDefaultDevice(MlxDevice dev);

    [LibraryImport("mlxc", EntryPoint = "mlx_device_is_available")]
    public static partial int IsAvailable([MarshalAs(UnmanagedType.U1)] out bool avail, MlxDevice dev);

    [LibraryImport("mlxc", EntryPoint = "mlx_device_count")]
    public static partial int Count(out int count, MlxDeviceType type);
}