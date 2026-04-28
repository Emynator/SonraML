using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Closure;
using SonraML.Backend.MLX.Interop.Map;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop.Io;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxFunctionExporter
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_function_exporter_new", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxFunctionExporter New
        (
        string file,
        MlxClosure fun,
        [MarshalAs(UnmanagedType.U1)] bool shapeless
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_function_exporter_free")]
    public static partial int Free(MlxFunctionExporter xfunc);

    [LibraryImport("mlxc", EntryPoint = "mlx_function_exporter_apply")]
    public static partial int Apply
        (
        MlxFunctionExporter xfunc,
        MlxVectorArray args
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_function_exporter_apply_kwargs")]
    public static partial int ApplyKwargs
        (
        MlxFunctionExporter xfunc,
        MlxVectorArray args,
        MlxMapStringToArray kwargs
        );
}