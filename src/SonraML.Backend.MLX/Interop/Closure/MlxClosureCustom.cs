using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop.Closure;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxClosureCustom
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_new")]
    public static partial MlxClosureCustom New();

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_new_func")]
    public static partial MlxClosureCustom NewFunc
        (delegate*<ref readonly MlxVectorArray, MlxVectorArray, MlxVectorArray, MlxVectorArray, int> fun);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_new_func_payload")]
    public static partial MlxClosureCustom NewFuncPayload
        (
        delegate*<ref readonly MlxVectorArray, MlxVectorArray, MlxVectorArray, MlxVectorArray, void*, int> fun,
        void* payload,
        delegate*<void*, void> dtor
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_free")]
    public static partial int Free(MlxClosureCustom cls);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_set")]
    public static partial int Set(ref readonly MlxClosureCustom cls, MlxClosureCustom src);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_apply")]
    public static partial int Apply
        (
        ref readonly MlxVectorArray res,
        MlxClosureCustom cls,
        MlxVectorArray input0,
        MlxVectorArray input1,
        MlxVectorArray input2
        );
}