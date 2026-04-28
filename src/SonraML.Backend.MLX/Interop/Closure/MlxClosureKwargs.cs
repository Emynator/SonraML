using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Map;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop.Closure;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxClosureKwargs
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_kwargs_new")]
    public static partial MlxClosureKwargs New();

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_kwargs_new_func")]
    public static partial MlxClosureKwargs NewFunc
        (delegate*<ref readonly MlxClosureKwargs, MlxClosureKwargs, MlxMapStringToArray, int> fun);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_kwargs_new_func_payload")]
    public static partial MlxClosureKwargs NewFuncPayload
        (
        delegate*<ref readonly MlxClosureKwargs, MlxClosureKwargs, MlxMapStringToArray, void*, int> fun,
        void* payload,
        delegate*<void*, void> dtor
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_kwargs_free")]
    public static partial int Free(MlxClosureKwargs cls);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_kwargs_set")]
    public static partial int Set(ref readonly MlxClosureKwargs cls, MlxClosureKwargs src);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_kwargs_apply")]
    public static partial int Apply
        (
        ref readonly MlxVectorArray res,
        MlxClosureKwargs cls,
        MlxVectorArray input0,
        MlxMapStringToArray input1
        );
}