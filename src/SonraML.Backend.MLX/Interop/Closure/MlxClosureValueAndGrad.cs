using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop.Closure;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxClosureValueAndGrad
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_value_and_grad_new")]
    public static partial MlxClosureValueAndGrad New();

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_value_and_grad_new_func")]
    public static partial MlxClosureValueAndGrad NewFunc
        (delegate*<ref readonly MlxVectorArray, ref readonly MlxVectorArray, MlxVectorArray, int> fun);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_value_and_grad_new_func_payload")]
    public static partial MlxClosureValueAndGrad NewFuncPayload
        (
        delegate*<ref readonly MlxVectorArray, ref readonly MlxVectorArray, MlxVectorArray, void*, int> fun,
        void* payload,
        delegate*<void*, void> dtor
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_value_and_grad_free")]
    public static partial int Free(MlxClosureValueAndGrad cls);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_value_and_grad_set")]
    public static partial int Set
        (
        MlxClosureValueAndGrad* cls,
        MlxClosureValueAndGrad src
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_value_and_grad_apply")]
    public static partial int Apply
        (
        MlxVectorArray* res0,
        MlxVectorArray* res1,
        MlxClosureValueAndGrad cls,
        MlxVectorArray input
        );
}