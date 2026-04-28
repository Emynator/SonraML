using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop.Map;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxMapStringToString
{
    private void* ctx;
    
    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_string_new")]
    public static partial MlxMapStringToString New();

    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_string_free")]
    public static partial int Free(MlxMapStringToString map);

    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_string_get", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Get(byte[][] value, MlxMapStringToString map, string key);

    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_string_set")]
    public static partial int Set(ref readonly MlxMapStringToString map, MlxMapStringToString src);

    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_string_insert", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Insert(MlxMapStringToString map, string key, string value);
}