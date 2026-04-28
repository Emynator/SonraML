using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop;

internal unsafe partial struct MlxNodeNamer
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_node_namer_new")]
    public static partial MlxNodeNamer New();

    [LibraryImport("mlxc", EntryPoint = "mlx_node_namer_free")]
    public static partial int Free(MlxNodeNamer namer);

    [LibraryImport("mlxc", EntryPoint = "mlx_node_namer_set_name", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SetName
        (
        MlxNodeNamer namer,
        MlxArray arr,
        string name
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_node_namer_get_name", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int GetName
        (
        string[] name,
        MlxNodeNamer namer,
        MlxArray arr
        );

    // [LibraryImport("mlxc", EntryPoint = "mlx_export_to_dot")]
    // public static partial int ExportToDot
    //     (
    //     FILE* os,
    //     MlxNodeNamer namer,
    //     MlxVectorArray outputs
    //     );
    //
    // [LibraryImport("mlxc", EntryPoint = "mlx_print_graph")]
    // public static partial int PrintGraph
    //     (
    //     FILE* os,
    //     MlxNodeNamer namer,
    //     MlxVectorArray outputs
    //     );
}