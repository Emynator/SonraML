using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Map;

namespace SonraML.Backend.MLX.Interop.Io;

internal static unsafe partial class MlxIo
{
    [LibraryImport("mlxc", EntryPoint = "mlx_load_reader")]
    public static partial int LoadReader
        (
        ref readonly MlxArray res,
        MlxIoReader inStream,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_load", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Load(ref readonly MlxArray res, string file, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_load_gguf", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int LoadGguf(ref readonly MlxIoGguf gguf, string file, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_load_safetensors_reader")]
    public static partial int LoadSafetensorsReader
        (
        ref readonly MlxMapStringToArray res0,
        ref readonly MlxMapStringToString res1,
        MlxIoReader inStream,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_load_safetensors", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int LoadSafetensors
        (
        ref readonly MlxMapStringToArray res0,
        ref readonly MlxMapStringToString res1,
        string file,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_save_writer")]
    public static partial int SaveWriter(MlxIoWriter outStream, MlxArray a);

    [LibraryImport("mlxc", EntryPoint = "mlx_save", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Save(string file, MlxArray a);

    [LibraryImport("mlxc", EntryPoint = "mlx_save_gguf", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SaveGguf(string file, MlxIoGguf gguf);

    [LibraryImport("mlxc", EntryPoint = "mlx_save_safetensors_writer")]
    public static partial int SaveSafetensorsWriter
        (
        MlxIoWriter inStream,
        MlxMapStringToArray param,
        MlxMapStringToString metadata
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_save_safetensors", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SaveSafetensors
        (
        string file,
        MlxMapStringToArray param,
        MlxMapStringToString metadata
        );
}