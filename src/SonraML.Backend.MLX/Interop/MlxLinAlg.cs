using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop;

internal static unsafe partial class MlxLinAlg
{
    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_cholesky")]
    public static partial int Cholesky
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool upper,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_cholesky_inv")]
    public static partial int CholeskyInv
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool upper,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_cross")]
    public static partial int Cross
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_eig")]
    public static partial int Eig
        (
        ref readonly MlxArray res0,
        ref readonly MlxArray res1,
        MlxArray a,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_eigh", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int EigH
        (
        ref readonly MlxArray res0,
        ref readonly MlxArray res1,
        MlxArray a,
        string uplO,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_eigvals")]
    public static partial int EigVals(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_eigvalsh", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int EigValsH
        (
        ref readonly MlxArray res,
        MlxArray a,
        string uplO,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_inv")]
    public static partial int Inv(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_lu")]
    public static partial int Lu(ref readonly MlxVectorArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_lu_factor")]
    public static partial int LuFactor
        (
        ref readonly MlxArray res0,
        ref readonly MlxArray res1,
        MlxArray a,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_norm")]
    public static partial int Norm
        (
        ref readonly MlxArray res,
        MlxArray a,
        double ord,
        int* axis,
        UIntPtr axisNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_norm_matrix", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int NormMatrix
        (
        ref readonly MlxArray res,
        MlxArray a,
        string ord,
        int* axis,
        UIntPtr axisNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_norm_l2")]
    public static partial int NormL2
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axis,
        UIntPtr axisNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_pinv")]
    public static partial int Pinv(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_qr")]
    public static partial int Qr
        (
        ref readonly MlxArray res0,
        ref readonly MlxArray res1,
        MlxArray a,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_solve")]
    public static partial int Solve
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_solve_triangular")]
    public static partial int SolveTriangular
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        [MarshalAs(UnmanagedType.U1)] bool upper,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_svd")]
    public static partial int Svd
        (
        ref readonly MlxVectorArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool computeUv,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linalg_tri_inv")]
    public static partial int TriInv
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool upper,
        MlxStream s
        );
}