using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Closure;
using SonraML.Backend.MLX.Interop.Enums;

namespace SonraML.Backend.MLX.Interop;

internal static unsafe partial class MlxCompile
{
    [LibraryImport("mlxc", EntryPoint = "mlx_compile")]
    public static partial int Compile
        (
        ref readonly MlxClosure res,
        MlxClosure fun,
        [MarshalAs(UnmanagedType.U1)] bool shapeless
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_detail_compile")]
    public static partial int DetailCompile
        (
        ref readonly MlxClosure res,
        MlxClosure fun,
        UIntPtr funId,
        [MarshalAs(UnmanagedType.U1)] bool shapeless,
        ulong* constants,
        UIntPtr constantsNum
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_detail_compile_clear_cache")]
    public static partial int DetailCompileClearCache();

    [LibraryImport("mlxc", EntryPoint = "mlx_detail_compile_erase")]
    public static partial int DetailCompileErase(UIntPtr funId);

    [LibraryImport("mlxc", EntryPoint = "mlx_disable_compile")]
    public static partial int DisableCompile();

    [LibraryImport("mlxc", EntryPoint = "mlx_enable_compile")]
    public static partial int EnableCompile();

    [LibraryImport("mlxc", EntryPoint = "mlx_set_compile_mode")]
    public static partial int SetCompileMode(CompileMode mode);
}