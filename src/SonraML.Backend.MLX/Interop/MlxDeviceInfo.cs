using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxDeviceInfo
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_device_info_new")]
    public static partial MlxDeviceInfo New();

    [LibraryImport("mlxc", EntryPoint = "mlx_device_info_get")]
    public static partial int Get(ref readonly MlxDeviceInfo info, MlxDevice dev);

    [LibraryImport("mlxc", EntryPoint = "mlx_device_info_free")]
    public static partial int Free(MlxDeviceInfo info);

    [LibraryImport("mlxc", EntryPoint = "mlx_device_info_has_key", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int HasKey
        (
        [MarshalAs(UnmanagedType.U1)] out bool exists,
        MlxDeviceInfo info,
        string key
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_device_info_is_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int IsString
        (
        [MarshalAs(UnmanagedType.U1)] out bool isString,
        MlxDeviceInfo info,
        string key
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_device_info_get_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int GetString
        (
        IntPtr* value,
        MlxDeviceInfo info,
        string key
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_device_info_get_size", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int GetSize
        (
        out UIntPtr value,
        MlxDeviceInfo info,
        string key
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_device_info_get_keys")]
    public static partial int GetKeys(ref readonly MlxVectorString keys, MlxDeviceInfo info);
}