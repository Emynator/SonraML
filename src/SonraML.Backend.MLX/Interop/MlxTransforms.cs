using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Closure;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop;

internal static unsafe partial class MlxTransforms
{
    [LibraryImport("mlxc", EntryPoint = "mlx_async_eval")]
    public static partial int AsyncEval(MlxVectorArray outputs);

    [LibraryImport("mlxc", EntryPoint = "mlx_checkpoint")]
    public static partial int Checkpoint(ref readonly MlxClosure res, MlxClosure fun);

    [LibraryImport("mlxc", EntryPoint = "mlx_custom_function")]
    public static partial int CustomFunction
        (
        ref readonly MlxClosure res,
        MlxClosure fun,
        MlxClosureCustom funVjp,
        MlxClosureCustomJvp funJvp,
        MlxClosureCustomVmap funVmap
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_custom_vjp")]
    public static partial int CustomVjp
        (
        ref readonly MlxClosure res,
        MlxClosure fun,
        MlxClosureCustom funVjp
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_eval")]
    public static partial int Eval(MlxVectorArray outputs);

    [LibraryImport("mlxc", EntryPoint = "mlx_jvp")]
    public static partial int Jvp
        (
        ref readonly MlxVectorArray res0,
        ref readonly MlxVectorArray res1,
        MlxClosure fun,
        MlxVectorArray primals,
        MlxVectorArray tangents
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_value_and_grad")]
    public static partial int ValueAndGrad
        (
        ref readonly MlxClosureValueAndGrad res,
        MlxClosure fun,
        int* argnums,
        UIntPtr argnumsNum
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_vjp")]
    public static partial int Vjp
        (
        ref readonly MlxVectorArray res0,
        ref readonly MlxVectorArray res1,
        MlxClosure fun,
        MlxVectorArray primals,
        MlxVectorArray cotangents
        );
}