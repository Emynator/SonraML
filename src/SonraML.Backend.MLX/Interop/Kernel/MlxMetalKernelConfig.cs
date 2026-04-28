using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Enums;

namespace SonraML.Backend.MLX.Interop.Kernel;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxMetalKernelConfig
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_config_new")]
    public static partial MlxMetalKernelConfig New();

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_config_free")]
    public static partial void Free(MlxMetalKernelConfig cls);

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_config_add_output_arg")]
    public static partial int AddOutputArg
        (
        MlxMetalKernelConfig cls,
        int* shape,
        UIntPtr size,
        DType dtype
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_config_set_grid")]
    public static partial int SetGrid
        (
        MlxMetalKernelConfig cls,
        int grid1,
        int grid2,
        int grid3
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_config_set_thread_group")]
    public static partial int SetThreadGroup
        (
        MlxMetalKernelConfig cls,
        int thread1,
        int thread2,
        int thread3
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_config_set_init_value")]
    public static partial int SetInitValue
        (
        MlxMetalKernelConfig cls,
        float value
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_config_set_verbose")]
    public static partial int SetVerbose
        (
        MlxMetalKernelConfig cls,
        [MarshalAs(UnmanagedType.U1)] bool verbose
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_config_add_template_arg_dtype", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int AddTemplateArgDType
        (
        MlxMetalKernelConfig cls,
        string name,
        DType dtype
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_config_add_template_arg_int", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int AddTemplateArgInt
        (
        MlxMetalKernelConfig cls,
        string name,
        int value
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fast_metal_kernel_config_add_template_arg_bool", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int AddTemplateArgBool
        (
        MlxMetalKernelConfig cls,
        string name,
        [MarshalAs(UnmanagedType.U1)] bool value
        );
}