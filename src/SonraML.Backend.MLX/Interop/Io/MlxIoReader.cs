using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop.Io;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxIoReader
{
    private void* ctx;
    
    [LibraryImport("mlxc", EntryPoint = "mlx_io_reader_new")]
    public static partial MlxIoReader New(void* desc, MlxIoVtable vtable);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_reader_free")]
    public static partial int Free(MlxIoReader io);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_reader_tostring")]
    public static partial int ToString(ref readonly MlxString str, MlxIoReader io);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_reader_descriptor")]
    public static partial int Descriptor(void** desc, MlxIoReader io);
}