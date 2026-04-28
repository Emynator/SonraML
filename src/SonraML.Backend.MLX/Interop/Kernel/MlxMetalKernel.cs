using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop.Kernel;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxMetalKernel
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_new", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxMetalKernel New
        (
        string name,
        MlxVectorString inputNames,
        MlxVectorString outputNames,
        string source,
        string header,
        [MarshalAs(UnmanagedType.U1)] bool ensureRowContiguous,
        [MarshalAs(UnmanagedType.U1)] bool atomicOutputs
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_free")]
    public static partial void Free(MlxMetalKernel cls);

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_apply")]
    public static partial int Apply
        (
        ref readonly MlxVectorArray outputs,
        MlxMetalKernel cls,
        MlxVectorArray inputs,
        MlxMetalKernelConfig config,
        MlxStream stream
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_rms_norm")]
    public static partial int RmsNorm
        (
        ref readonly MlxArray res,
        MlxArray x,
        MlxArray weight,
        float eps,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_rope")]
    public static partial int Rope
        (
        ref readonly MlxArray res,
        MlxArray x,
        int dims,
        [MarshalAs(UnmanagedType.U1)] bool traditional,
        MlxOptionalFloat bse,
        float scale,
        int offset,
        MlxArray freqs,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_rope_dynamic")]
    public static partial int RopeDynamic
        (
        ref readonly MlxArray res,
        MlxArray x,
        int dims,
        [MarshalAs(UnmanagedType.U1)] bool traditional,
        MlxOptionalFloat bse,
        float scale,
        MlxArray offset,
        MlxArray freqs,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_scaled_dot_product_attention", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int ScaledDotProductAttention
        (
        ref readonly MlxArray res,
        MlxArray queries,
        MlxArray keys,
        MlxArray values,
        float scale,
        string maskMode,
        MlxArray maskArr,
        MlxArray sinks,
        MlxStream s
        );
}