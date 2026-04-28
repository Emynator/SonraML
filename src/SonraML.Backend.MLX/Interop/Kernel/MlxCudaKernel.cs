using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop.Kernel;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxCudaKernel
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_new", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxCudaKernel New
        (
        string name,
        MlxVectorString inputNames,
        MlxVectorString outputNames,
        string source,
        string header,
        [MarshalAs(UnmanagedType.U1)] bool ensureRowContiguous,
        int sharedMemory
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_free")]
    public static partial void Free(MlxCudaKernel cls);

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_apply")]
    public static partial int Apply
        (
        ref readonly MlxVectorArray outputs,
        MlxCudaKernel cls,
        MlxVectorArray inputs,
        MlxCudaKernelConfig config,
        MlxStream stream
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_layer_norm")]
    public static partial int LayerNorm
        (
        ref readonly MlxArray res,
        MlxArray x,
        MlxArray weight,
        MlxArray bias,
        float eps,
        MlxStream s
        );
}