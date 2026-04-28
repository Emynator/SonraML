using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop.Closure;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxClosureCustomVmap
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_vmap_new")]
    public static partial MlxClosureCustomVmap New();

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_vmap_new_func")]
    public static partial MlxClosureCustomVmap NewFunc
        (delegate*<ref readonly MlxVectorArray, ref readonly MlxVectorInt, MlxVectorArray, int*, UIntPtr, int> fun);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_vmap_new_func_payload")]
    public static partial MlxClosureCustomVmap NewFuncPayload
        (
        delegate*<ref readonly MlxVectorArray, ref readonly MlxVectorInt, MlxVectorArray, int*, UIntPtr, int> fun,
        void* payload,
        delegate*<void*, void> dtor
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_vmap_free")]
    public static partial int Free(MlxClosureCustomVmap cls);

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_vmap_set")]
    public static partial int Set
        (
        ref readonly MlxClosureCustomVmap cls,
        MlxClosureCustomVmap src
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_closure_custom_vmap_apply")]
    public static partial int Apply
        (
        ref readonly MlxVectorArray res0,
        ref readonly MlxVectorInt res1,
        MlxClosureCustomVmap cls,
        MlxVectorArray input0,
        int* input1,
        UIntPtr input1Num
        );
}