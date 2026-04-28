using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop.Closure;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxClosure
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_new")]
    public static partial MlxClosure New();

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_new_func")]
    public static partial MlxClosure NewFunc(delegate*<ref readonly MlxVectorArray, MlxVectorArray, int> fun);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_new_func_payload")]
    public static partial MlxClosure NewFuncPayload
        (
        delegate*<ref readonly MlxVectorArray, MlxVectorArray, void*, int> fun,
        void* payload,
        delegate*<void*, void> dtor
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_new_unary")]
    public static partial MlxClosure NewUnary(delegate*<ref readonly MlxArray, MlxArray, int> fun);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_free")]
    public static partial int Free(MlxClosure cls);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_set")]
    public static partial int Set(ref readonly MlxClosure cls, MlxClosure src);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_apply")]
    public static partial int Apply
        (
        MlxVectorArray* res,
        MlxClosure cls,
        MlxVectorArray input
        );
}