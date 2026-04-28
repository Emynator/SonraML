using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Closure;
using SonraML.Backend.MLX.Interop.Map;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop;

internal static unsafe partial class MlxExport
{
    [LibraryImport("mlxc", EntryPoint = "mlx_export_function", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Function
        (
        string file,
        MlxClosure fun,
        MlxVectorArray args,
        [MarshalAs(UnmanagedType.U1)] bool shapeless
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_export_function_kwargs", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int FunctionKwargs
        (
        string file,
        MlxClosureKwargs fun,
        MlxVectorArray args,
        MlxMapStringToArray kwargs,
        [MarshalAs(UnmanagedType.U1)] bool shapeless
        );
}