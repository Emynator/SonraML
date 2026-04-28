using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Map;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop.Io;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxImportedFunction
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_imported_function_new", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxImportedFunction New(string file);

    [LibraryImport("mlxc", EntryPoint = "mlx_imported_function_free")]
    public static partial int Free(MlxImportedFunction xfunc);

    [LibraryImport("mlxc", EntryPoint = "mlx_imported_function_apply")]
    public static partial int Apply
        (
        MlxVectorArray* res,
        MlxImportedFunction xfunc,
        MlxVectorArray args
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_imported_function_apply_kwargs")]
    public static partial int ApplyKwargs
        (
        MlxVectorArray* res,
        MlxImportedFunction xfunc,
        MlxVectorArray args,
        MlxMapStringToArray kwargs
        );
}