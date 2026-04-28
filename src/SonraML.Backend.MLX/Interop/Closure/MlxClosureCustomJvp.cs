using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop.Closure;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxClosureCustomJvp
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_jvp_new")]
    public static partial MlxClosureCustomJvp New();

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_jvp_new_func")]
    public static partial MlxClosureCustomJvp NewFunc
        (delegate*<ref readonly MlxVectorArray, MlxVectorArray, MlxVectorArray, int*, UIntPtr, int> fun);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_jvp_new_func_payload")]
    public static partial MlxClosureCustomJvp NewFuncPayload
        (
        delegate*<ref readonly MlxVectorArray, MlxVectorArray, MlxVectorArray, int*, UIntPtr, void*, int> fun,
        void* payload,
        delegate*<void*, void> dtor
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_jvp_free")]
    public static partial int Free(MlxClosureCustomJvp cls);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_jvp_set")]
    public static partial int Set(ref readonly MlxClosureCustomJvp cls, MlxClosureCustomJvp src);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_jvp_apply")]
    public static partial int Apply
        (
        ref readonly MlxVectorArray res,
        MlxClosureCustomJvp cls,
        MlxVectorArray input0,
        MlxVectorArray input1,
        int* input2,
        UIntPtr input2Num
        );
}