using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxStream
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_stream_new")]
    public static partial MlxStream New();

    [LibraryImport("mlxc", EntryPoint = "mlx_stream_new_device")]
    public static partial MlxStream NewDevice(MlxDevice dev);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_stream_set")]
    public static partial int Set(ref MlxStream stream, MlxStream src);
        
    [LibraryImport("mlxc", EntryPoint = "mlx_stream_free")]
    public static partial int Free(MlxStream stream);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_stream_tostring")]
    public static partial int ToString(ref MlxString str, MlxStream stream);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_stream_equal")]
    [return: MarshalAs(UnmanagedType.I1)]
    public static partial bool Equal(MlxStream lhs, MlxStream rhs);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_stream_get_device")]
    public static partial int GetDevice(out MlxDevice dev, MlxStream stream);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_stream_get_index")]
    public static partial int GetIndex(out int index, MlxStream stream);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_synchronize")]
    public static partial int Synchronize(MlxStream stream);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_get_default_stream")]
    public static partial int GetDefaultStream(out MlxStream stream, MlxDevice dev);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_set_default_stream")]
    public static partial int SetDefaultStream(MlxStream stream);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_default_cpu_stream_new")]
    public static partial MlxStream DefaultCpuStreamNew();

    [LibraryImport("mlxc", EntryPoint = "mlx_default_gpu_stream_new")]
    public static partial MlxStream DefaultGpuStreamNew();
}