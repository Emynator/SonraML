using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop.Map;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxMapStringToArray
{
    private void* ctx;
    
    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_array_new")]
    public static partial MlxMapStringToArray New();

    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_array_free")]
    public static partial int Free(MlxMapStringToArray map);

    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_array_get", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Get(ref readonly MlxArray value, MlxMapStringToArray map, string key);

    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_array_set")]
    public static partial int Set(ref readonly MlxMapStringToArray map, MlxMapStringToArray src);

    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_array_insert", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Insert(MlxMapStringToArray map, string key, MlxArray value);
}