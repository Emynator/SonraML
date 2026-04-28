using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxString
{
    private void* ctx;
    
    [LibraryImport("mlxc", EntryPoint = "mlx_string_new")]
    public static partial MlxString New();

    [LibraryImport("mlxc", EntryPoint = "mlx_string_new_data", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxString NewData(string str);

    [LibraryImport("mlxc", EntryPoint = "mlx_string_free")]
    public static partial int Free(MlxString str);

    [LibraryImport("mlxc", EntryPoint = "mlx_string_set")]
    public static partial int Set(ref MlxString str, MlxString src);

    [LibraryImport("mlxc", EntryPoint = "mlx_string_data", StringMarshalling = StringMarshalling.Utf8)]
    public static partial string Data(MlxString str);
}