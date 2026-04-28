using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Enums;

namespace SonraML.Backend.MLX.Interop.Kernel;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxCudaKernelConfig
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_config_new")]
    public static partial MlxCudaKernelConfig New();

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_config_free")]
    public static partial void Free(MlxCudaKernelConfig cls);

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_config_add_output_arg")]
    public static partial int AddOutputArg
        (
        MlxCudaKernelConfig cls,
        int* shape,
        UIntPtr size,
        DType dtype
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_config_set_grid")]
    public static partial int SetGrid
        (
        MlxCudaKernelConfig cls,
        int grid1,
        int grid2,
        int grid3
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_config_set_thread_group")]
    public static partial int SetThreadGroup
        (
        MlxCudaKernelConfig cls,
        int thread1,
        int thread2,
        int thread3
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_config_set_init_value")]
    public static partial int SetInitValue
        (
        MlxCudaKernelConfig cls,
        float value
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_config_set_verbose")]
    public static partial int SetVerbose
        (
        MlxCudaKernelConfig cls,
        [MarshalAs(UnmanagedType.U1)] bool verbose
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_config_add_template_arg_dtype", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int AddTemplateArgDType
        (
        MlxCudaKernelConfig cls,
        string name,
        DType dtype
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_config_add_template_arg_int", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int AddTemplateArgInt
        (
        MlxCudaKernelConfig cls,
        string name,
        int value
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_cuda_kernel_config_add_template_arg_bool", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int AddTemplateArgBool
        (
        MlxCudaKernelConfig cls,
        string name,
        [MarshalAs(UnmanagedType.U1)] bool value
        );
}