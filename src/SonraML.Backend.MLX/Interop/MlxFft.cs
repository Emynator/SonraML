using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Enums;

namespace SonraML.Backend.MLX.Interop;

internal static unsafe partial class MlxFft
{
    [LibraryImport("mlxc", EntryPoint = "mlx_fft_fft")]
    public static partial int Fft
        (
        ref readonly MlxArray res,
        MlxArray a,
        int n,
        int axis,
        FftNorm norm,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_fft2")]
    public static partial int Fft2
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* n,
        UIntPtr nNum,
        int* axes,
        UIntPtr axesNum,
        FftNorm norm,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_fftfreq")]
    public static partial int FftFreq(ref readonly MlxArray res, int n, double d, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_fftn")]
    public static partial int FftN
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* n,
        UIntPtr nNum,
        int* axes,
        UIntPtr axesNum,
        FftNorm norm,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_fftshift")]
    public static partial int FftShift
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_ifft")]
    public static partial int InvFft
        (
        ref readonly MlxArray res,
        MlxArray a,
        int n,
        int axis,
        FftNorm norm,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_ifft2")]
    public static partial int InvFft2
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* n,
        UIntPtr nNum,
        int* axes,
        UIntPtr axesNum,
        FftNorm norm,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_ifftn")]
    public static partial int InvFftN
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* n,
        UIntPtr nNum,
        int* axes,
        UIntPtr axesNum,
        FftNorm norm,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_ifftshift")]
    public static partial int InvFftShift
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_irfft")]
    public static partial int InvRealFft
        (
        ref readonly MlxArray res,
        MlxArray a,
        int n,
        int axis,
        FftNorm norm,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_irfft2")]
    public static partial int InvRealFft2
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* n,
        UIntPtr nNum,
        int* axes,
        UIntPtr axesNum,
        FftNorm norm,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_irfftn")]
    public static partial int InvRealFftN
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* n,
        UIntPtr nNum,
        int* axes,
        UIntPtr axesNum,
        FftNorm norm,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_rfft")]
    public static partial int RealFft
        (
        ref readonly MlxArray res,
        MlxArray a,
        int n,
        int axis,
        FftNorm norm,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_rfft2")]
    public static partial int RealFft2
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* n,
        UIntPtr nNum,
        int* axes,
        UIntPtr axesNum,
        FftNorm norm,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_rfftfreq")]
    public static partial int RealFftFreq(ref readonly MlxArray res, int n, double d, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_fft_rfftn")]
    public static partial int RealFftN
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* n,
        UIntPtr nNum,
        int* axes,
        UIntPtr axesNum,
        FftNorm norm,
        MlxStream s
        );
}