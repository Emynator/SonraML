using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop;

internal static unsafe partial class Mlx
{
    [LibraryImport("mlxc", EntryPoint = "mlx_version")]
    public static partial int Version(ref readonly MlxString str);

    [LibraryImport("mlxc", EntryPoint = "mlx_set_error_handler")]
    public static partial void SetErrorHandler
        (
        delegate*<byte*, void*, void> handler,
        void* data,
        delegate*<void*, void> dtor
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_cuda_is_available")]
    public static partial int CudaIsAvailable(byte* res);

    [LibraryImport("mlxc", EntryPoint = "mlx_metal_is_available")]
    public static partial int MetalIsAvailable(byte* res);

    [LibraryImport("mlxc", EntryPoint = "mlx_metal_start_capture", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int MetalStartCapture(string path);

    [LibraryImport("mlxc", EntryPoint = "mlx_metal_stop_capture")]
    public static partial int MetalStopCapture();
}