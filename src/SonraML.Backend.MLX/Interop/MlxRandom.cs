using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Enums;

namespace SonraML.Backend.MLX.Interop;

internal static unsafe partial class MlxRandom
{
    [LibraryImport("mlxc", EntryPoint = "mlx_random_bernoulli")]
    public static partial int Bernoulli
        (
        ref readonly MlxArray res,
        MlxArray p,
        int* shape,
        UIntPtr shapeNum,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_bits")]
    public static partial int Bits
        (
        ref readonly MlxArray res,
        int* shape,
        UIntPtr shapeNum,
        int width,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_categorical_shape")]
    public static partial int Categorical
        (
        ref readonly MlxArray res,
        MlxArray logits,
        int axis,
        int* shape,
        UIntPtr shapeNum,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_categorical_num_samples")]
    public static partial int Categorical
        (
        ref readonly MlxArray res,
        MlxArray logits,
        int axis,
        int numSamples,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_categorical")]
    public static partial int Categorical
        (
        ref readonly MlxArray res,
        MlxArray logits,
        int axis,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_gumbel")]
    public static partial int Gumbel
        (
        ref readonly MlxArray res,
        int* shape,
        UIntPtr shapeNum,
        DType dtype,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_key")]
    public static partial int Key(ref readonly MlxArray res, ulong seed);

    [LibraryImport("mlxc", EntryPoint = "mlx_random_laplace")]
    public static partial int Laplace
        (
        ref readonly MlxArray res,
        int* shape,
        UIntPtr shapeNum,
        DType dtype,
        float loc,
        float scale,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_multivariate_normal")]
    public static partial int MultivariateNormal
        (
        ref readonly MlxArray res,
        MlxArray mean,
        MlxArray cov,
        int* shape,
        UIntPtr shapeNum,
        DType dtype,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_normal_broadcast")]
    public static partial int NormalBroadcast
        (
        ref readonly MlxArray res,
        int* shape,
        UIntPtr shapeNum,
        DType dtype,
        MlxArray loc,
        MlxArray scale,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_normal")]
    public static partial int Normal
        (
        ref readonly MlxArray res,
        int* shape,
        UIntPtr shapeNum,
        DType dtype,
        float loc,
        float scale,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_permutation")]
    public static partial int Permutation
        (
        ref readonly MlxArray res,
        MlxArray x,
        int axis,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_permutation_arange")]
    public static partial int PermutationArange
        (
        ref readonly MlxArray res,
        int x,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_randint")]
    public static partial int RandInt
        (
        ref readonly MlxArray res,
        MlxArray low,
        MlxArray high,
        int* shape,
        UIntPtr shapeNum,
        DType dtype,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_seed")]
    public static partial int Seed(ulong seed);

    [LibraryImport("mlxc", EntryPoint = "mlx_random_split_num")]
    public static partial int SplitNum
        (
        ref readonly MlxArray res,
        MlxArray key,
        int num,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_split")]
    public static partial int Split
        (
        ref readonly MlxArray res0,
        ref readonly MlxArray res1,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_truncated_normal")]
    public static partial int TruncatedNormal
        (
        ref readonly MlxArray res,
        MlxArray lower,
        MlxArray upper,
        int* shape,
        UIntPtr shapeNum,
        DType dtype,
        MlxArray key,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_random_uniform")]
    public static partial int Uniform
        (
        ref readonly MlxArray res,
        MlxArray low,
        MlxArray high,
        int* shape,
        UIntPtr shapeNum,
        DType dtype,
        MlxArray key,
        MlxStream s
        );
}