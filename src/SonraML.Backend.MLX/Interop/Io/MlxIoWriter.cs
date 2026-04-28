using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop.Io;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxIoWriter
{
    private void* ctx;
    
    [LibraryImport("mlxc", EntryPoint = "mlx_io_writer_new")]
    public static partial MlxIoWriter New(void* desc, MlxIoVtable vtable);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_writer_free")]
    public static partial int Free(MlxIoWriter io);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_writer_tostring")]
    public static partial int ToString(ref readonly MlxString str, MlxIoWriter io);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_writer_descriptor")]
    public static partial int Descriptor(void** desc, MlxIoWriter io);
}