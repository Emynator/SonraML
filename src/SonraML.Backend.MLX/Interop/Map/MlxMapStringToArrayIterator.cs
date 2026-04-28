using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop.Map;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxMapStringToArrayIterator
{
    private void* ctx;
    
    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_array_iterator_new")]
    public static partial MlxMapStringToArrayIterator New(MlxMapStringToArray map);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_array_iterator_free")]
    public static partial int Free(MlxMapStringToArrayIterator it);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_map_string_to_array_iterator_next", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Next(IntPtr* key, ref readonly MlxArray value, MlxMapStringToArrayIterator it);
}