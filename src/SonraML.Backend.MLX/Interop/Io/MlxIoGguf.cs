using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop.Io;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxIoGguf
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_new")]
    public static partial MlxIoGguf New();

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_free")]
    public static partial int Free(MlxIoGguf io);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_get_keys")]
    public static partial int GetKeys(ref readonly MlxVectorString keys, MlxIoGguf io);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_has_metadata_array", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int HasMetadataArray(ref readonly byte flag, MlxIoGguf io, string key);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_get_metadata_array", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int GetMetadataArray(ref readonly MlxArray arr, MlxIoGguf io, string key);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_set_metadata_array", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SetMetadataArray(MlxIoGguf io, string key, MlxArray marr);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_has_metadata_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int HasMetadataString(ref readonly byte flag, MlxIoGguf io, string key);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_get_metadata_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int GetMetadataString(ref readonly MlxString str, MlxIoGguf io, string key);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_set_metadata_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SetMetadataString(MlxIoGguf io, string key, string mstr);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_has_metadata_vector_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int HasMetadataVectorString(ref readonly byte flag, MlxIoGguf io, string key);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_get_metadata_vector_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int GetMetadataVectorString(ref readonly MlxVectorString vstr, MlxIoGguf io, string key);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_set_metadata_vector_string", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SetMetadataVectorString(MlxIoGguf io, string key, MlxVectorString mvstr);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_get_array", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int GetArray(ref readonly MlxArray arr, MlxIoGguf io, string key);

    [LibraryImport("mlxc", EntryPoint = "mlx_io_gguf_set_array", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SetArray(MlxIoGguf io, string key, MlxArray arr);
}