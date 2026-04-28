using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop.Map;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxMapStringToStringIterator
{
    private void* ctx;
    
    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_string_iterator_new")]
    public static partial MlxMapStringToStringIterator New(MlxMapStringToString map);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_string_iterator_free")]
    public static partial int Free(MlxMapStringToStringIterator it);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_string_iterator_next", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Next(string[] key, byte[][] value, MlxMapStringToStringIterator it);
}