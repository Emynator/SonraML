using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Interop;

internal static unsafe partial class MlxOps
{
    [LibraryImport("mlxc", EntryPoint = "mlx_abs")]
    public static partial int Abs(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_add")]
    public static partial int Add
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_addmm")]
    public static partial int AddMM
        (
        ref readonly MlxArray res,
        MlxArray c,
        MlxArray a,
        MlxArray b,
        float alpha,
        float beta,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_all_axes")]
    public static partial int AllAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_all_axis")]
    public static partial int AllAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_all")]
    public static partial int All
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_allclose")]
    public static partial int AllClose
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        double rtol,
        double atol,
        [MarshalAs(UnmanagedType.U1)] bool equalNAN,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_any_axes")]
    public static partial int AnyAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_any_axis")]
    public static partial int AnyAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_any")]
    public static partial int Any
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_arange")]
    public static partial int Arange
        (
        ref readonly MlxArray res,
        double start,
        double stop,
        double step,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_arccos")]
    public static partial int ArcCos(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_arccosh")]
    public static partial int ArcCosH(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_arcsin")]
    public static partial int ArcSin(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_arcsinh")]
    public static partial int ArcSinH(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_arctan")]
    public static partial int ArcTan(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_arctan2")]
    public static partial int ArcTan2
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_arctanh")]
    public static partial int ArcTanH(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_argmax_axis")]
    public static partial int ArgMaxAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_argmax")]
    public static partial int ArgMax
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_argmin_axis")]
    public static partial int ArgMinAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_argmin")]
    public static partial int ArgMin
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_argpartition_axis")]
    public static partial int ArgPartitionAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int kth,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_argpartition")]
    public static partial int ArgPartition
        (
        ref readonly MlxArray res,
        MlxArray a,
        int kth,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_argsort_axis")]
    public static partial int ArgSortAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_argsort")]
    public static partial int ArgSort(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_equal")]
    public static partial int ArrayEqual
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        [MarshalAs(UnmanagedType.U1)] bool equalNAN,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_as_strided")]
    public static partial int AsStrided
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* shape,
        UIntPtr shapeNum,
        long* strides,
        UIntPtr stridesNum,
        UIntPtr offset,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_astype")]
    public static partial int AsType
        (
        ref readonly MlxArray res,
        MlxArray a,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_atleast_1d")]
    public static partial int AtLeast1d(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_atleast_2d")]
    public static partial int AtLeast2d(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_atleast_3d")]
    public static partial int AtLeast3d(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_bartlett")]
    public static partial int Bartlett(ref readonly MlxArray res, int m, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_bitwise_and")]
    public static partial int BitwiseAnd
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_bitwise_invert")]
    public static partial int BitwiseInvert(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_bitwise_or")]
    public static partial int BitwiseOr
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_bitwise_xor")]
    public static partial int BitwiseXor
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_blackman")]
    public static partial int Blackman(ref readonly MlxArray res, int m, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_block_masked_mm")]
    public static partial int BlockMaskedMM
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        int blockSize,
        MlxArray maskOut,
        MlxArray maskLhs,
        MlxArray maskRhs,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_broadcast_arrays")]
    public static partial int BroadcastArrays
        (
        ref readonly MlxVectorArray res,
        MlxVectorArray inputs,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_broadcast_to")]
    public static partial int BroadcastTo
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* shape,
        UIntPtr shapeNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_ceil")]
    public static partial int Ceil(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_clip")]
    public static partial int Clip
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray aMin,
        MlxArray aMax,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_concatenate_axis")]
    public static partial int ConcatenateAxis
        (
        ref readonly MlxArray res,
        MlxVectorArray arrays,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_concatenate")]
    public static partial int Concatenate
        (
        ref readonly MlxArray res,
        MlxVectorArray arrays,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_conjugate")]
    public static partial int Conjugate(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_contiguous")]
    public static partial int Contiguous
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool allowColMajor,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_conv1d")]
    public static partial int Conv1d
        (
        ref readonly MlxArray res,
        MlxArray input,
        MlxArray weight,
        int stride,
        int padding,
        int dilation,
        int groups,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_conv2d")]
    public static partial int Conv2d
        (
        ref readonly MlxArray res,
        MlxArray input,
        MlxArray weight,
        int stride0,
        int stride1,
        int padding0,
        int padding1,
        int dilation0,
        int dilation1,
        int groups,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_conv3d")]
    public static partial int Conv3d
        (
        ref readonly MlxArray res,
        MlxArray input,
        MlxArray weight,
        int stride0,
        int stride1,
        int stride2,
        int padding0,
        int padding1,
        int padding2,
        int dilation0,
        int dilation1,
        int dilation2,
        int groups,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_conv_general")]
    public static partial int ConvGeneral
        (
        ref readonly MlxArray res,
        MlxArray input,
        MlxArray weight,
        int* stride,
        UIntPtr strideNum,
        int* paddingLo,
        UIntPtr paddingLoNum,
        int* paddingHi,
        UIntPtr paddingHiNum,
        int* kernelDilation,
        UIntPtr kernelDilationNum,
        int* inputDilation,
        UIntPtr inputDilationNum,
        int groups,
        [MarshalAs(UnmanagedType.U1)] bool flip,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_conv_transpose1d")]
    public static partial int ConvTranspose1d
        (
        ref readonly MlxArray res,
        MlxArray input,
        MlxArray weight,
        int stride,
        int padding,
        int dilation,
        int outputPadding,
        int groups,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_conv_transpose2d")]
    public static partial int ConvTranspose2d
        (
        ref readonly MlxArray res,
        MlxArray input,
        MlxArray weight,
        int stride0,
        int stride1,
        int padding0,
        int padding1,
        int dilation0,
        int dilation1,
        int outputPadding0,
        int outputPadding1,
        int groups,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_conv_transpose3d")]
    public static partial int ConvTranspose3d
        (
        ref readonly MlxArray res,
        MlxArray input,
        MlxArray weight,
        int stride0,
        int stride1,
        int stride2,
        int padding0,
        int padding1,
        int padding2,
        int dilation0,
        int dilation1,
        int dilation2,
        int outputPadding0,
        int outputPadding1,
        int outputPadding2,
        int groups,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_copy")]
    public static partial int Copy(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_cos")]
    public static partial int Cos(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_cosh")]
    public static partial int CosH(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_cummax")]
    public static partial int CumMax
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool reverse,
        [MarshalAs(UnmanagedType.U1)] bool inclusive,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_cummin")]
    public static partial int CumMin
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool reverse,
        [MarshalAs(UnmanagedType.U1)] bool inclusive,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_cumprod")]
    public static partial int CumProd
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool reverse,
        [MarshalAs(UnmanagedType.U1)] bool inclusive,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_cumsum")]
    public static partial int CumSum
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool reverse,
        [MarshalAs(UnmanagedType.U1)] bool inclusive,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_degrees")]
    public static partial int Degrees(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_depends")]
    public static partial int Depends
        (
        ref readonly MlxVectorArray res,
        MlxVectorArray inputs,
        MlxVectorArray dependencies
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_dequantize", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Dequantize
        (
        ref readonly MlxArray res,
        MlxArray w,
        MlxArray scales,
        MlxArray biases /* may be null */,
        MlxOptionalInt groupSize,
        MlxOptionalInt bits,
        string mode,
        MlxArray globalScale /* may be null */,
        MlxOptionalDType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_diag")]
    public static partial int Diag(ref readonly MlxArray res, MlxArray a, int k, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_diagonal")]
    public static partial int Diagonal
        (
        ref readonly MlxArray res,
        MlxArray a,
        int offset,
        int axis1,
        int axis2,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_divide")]
    public static partial int Divide
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_divmod")]
    public static partial int DivMod
        (
        ref readonly MlxVectorArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_einsum", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int EinSum
        (
        ref readonly MlxArray res,
        string subscripts,
        MlxVectorArray operands,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_equal")]
    public static partial int Equal
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_erf")]
    public static partial int Erf(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_erfinv")]
    public static partial int ErfInv(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_exp")]
    public static partial int Exp(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_expand_dims_axes")]
    public static partial int ExpandDimsAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_expand_dims")]
    public static partial int ExpandDims
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_expm1")]
    public static partial int ExpM1(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_eye")]
    public static partial int Eye
        (
        ref readonly MlxArray res,
        int n,
        int m,
        int k,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_flatten")]
    public static partial int Flatten
        (
        ref readonly MlxArray res,
        MlxArray a,
        int startAxis,
        int endAxis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_floor")]
    public static partial int Floor(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_floor_divide")]
    public static partial int FloorDivide
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_from_fp8")]
    public static partial int FromF8
        (
        ref readonly MlxArray res,
        MlxArray x,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_full")]
    public static partial int Full
        (
        ref readonly MlxArray res,
        int* shape,
        UIntPtr shapeNum,
        MlxArray vals,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_full_like")]
    public static partial int FullLike
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray vals,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_gather")]
    public static partial int Gather
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxVectorArray indices,
        int* axes,
        UIntPtr axesNum,
        int* sliceSizes,
        UIntPtr sliceSizesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_gather_single")]
    public static partial int GatherSingle
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray indices,
        int axis,
        int* sliceSizes,
        UIntPtr sliceSizesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_gather_mm")]
    public static partial int GatherMM
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxArray lhsIndices,
        MlxArray rhsIndices,
        [MarshalAs(UnmanagedType.U1)] bool sortedIndices,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_gather_qmm", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int GatherQMM
        (
        ref readonly MlxArray res,
        MlxArray x,
        MlxArray w,
        MlxArray scales,
        MlxArray biases,
        MlxArray lhsIndices,
        MlxArray rhsIndices,
        [MarshalAs(UnmanagedType.U1)] bool transpose,
        MlxOptionalInt groupSize,
        MlxOptionalInt bits,
        string mode,
        [MarshalAs(UnmanagedType.U1)] bool sortedIndices,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_greater")]
    public static partial int Greater
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_greater_equal")]
    public static partial int GreaterEqual
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_hadamard_transform")]
    public static partial int HadamardTransform
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxOptionalFloat scale,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_hamming")]
    public static partial int Hamming(ref readonly MlxArray res, int m, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_hanning")]
    public static partial int Hanning(ref readonly MlxArray res, int m, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_identity")]
    public static partial int Identity(ref readonly MlxArray res, int n, DType dtype, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_imag")]
    public static partial int Imag(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_inner")]
    public static partial int Inner
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_isclose")]
    public static partial int IsClose
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        double rtol,
        double atol,
        [MarshalAs(UnmanagedType.U1)] bool equalNAN,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_isfinite")]
    public static partial int IsFinite(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_isinf")]
    public static partial int IsInf(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_isnan")]
    public static partial int IsNAN(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_isneginf")]
    public static partial int IsNegInf(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_isposinf")]
    public static partial int IsPosInf(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_kron")]
    public static partial int Kron
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_left_shift")]
    public static partial int LeftShift
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_less")]
    public static partial int Less
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_less_equal")]
    public static partial int LessEqual
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_linspace")]
    public static partial int Linspace
        (
        ref readonly MlxArray res,
        double start,
        double stop,
        int num,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_log")]
    public static partial int Log(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_log10")]
    public static partial int Log10(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_log1p")]
    public static partial int Log1P(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_log2")]
    public static partial int Log2(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_logaddexp")]
    public static partial int LogAddExp
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_logcumsumexp")]
    public static partial int LogCumSumExp
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool reverse,
        [MarshalAs(UnmanagedType.U1)] bool inclusive,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_logical_and")]
    public static partial int LogicalAnd
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_logical_not")]
    public static partial int LogicalNot(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_logical_or")]
    public static partial int LogicalOr
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_logsumexp_axes")]
    public static partial int LogSumExpAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_logsumexp_axis")]
    public static partial int LogSumExpAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_logsumexp")]
    public static partial int LogSumExp
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_masked_scatter")]
    public static partial int MaskedScatter
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray mask,
        MlxArray src,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_matmul")]
    public static partial int MatMul
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_max_axes")]
    public static partial int MaxAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_max_axis")]
    public static partial int MaxAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_max")]
    public static partial int Max
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_maximum")]
    public static partial int Maximum
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_mean_axes")]
    public static partial int MeanAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_mean_axis")]
    public static partial int MeanAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_mean")]
    public static partial int Mean
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_median")]
    public static partial int Median
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_meshgrid", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int MeshGrid
        (
        ref readonly MlxVectorArray res,
        MlxVectorArray arrays,
        [MarshalAs(UnmanagedType.U1)] bool sparse,
        string indexing,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_min_axes")]
    public static partial int MinAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_min_axis")]
    public static partial int MinAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_min")]
    public static partial int Min
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_minimum")]
    public static partial int Minimum
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_moveaxis")]
    public static partial int MoveAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int source,
        int destination,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_multiply")]
    public static partial int Multiply
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_nan_to_num")]
    public static partial int NANToNum
        (
        ref readonly MlxArray res,
        MlxArray a,
        float nan,
        MlxOptionalFloat posInf,
        MlxOptionalFloat negInf,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_negative")]
    public static partial int Negative(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_not_equal")]
    public static partial int NotEqual
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_number_of_elements")]
    public static partial int NumberOfElements
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool inverted,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_ones")]
    public static partial int Ones
        (
        ref readonly MlxArray res,
        int* shape,
        UIntPtr shapeNum,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_ones_like")]
    public static partial int OnesLike(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_outer")]
    public static partial int Outer
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_pad", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Pad
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        int* lowPadSize,
        UIntPtr lowPadSizeNum,
        int* highPadSize,
        UIntPtr highPadSizeNum,
        MlxArray padValue,
        string mode,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_pad_symmetric", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int PadSymmetric
        (
        ref readonly MlxArray res,
        MlxArray a,
        int padWidth,
        MlxArray padValue,
        string mode,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_partition_axis")]
    public static partial int PartitionAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int kth,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_partition")]
    public static partial int Partition
        (
        ref readonly MlxArray res,
        MlxArray a,
        int kth,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_power")]
    public static partial int Power
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_prod_axes")]
    public static partial int ProdAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_prod_axis")]
    public static partial int ProdAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_prod")]
    public static partial int Prod
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_put_along_axis")]
    public static partial int PutAlongAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray indices,
        MlxArray values,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_qqmm", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int QQMM
        (
        ref readonly MlxArray res,
        MlxArray x,
        MlxArray w,
        MlxArray wScales,
        MlxOptionalInt groupSize,
        MlxOptionalInt bits,
        string mode,
        MlxArray globalScaleX,
        MlxArray globalScaleW,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_quantize", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Quantize
        (
        ref readonly MlxVectorArray res,
        MlxArray w,
        MlxOptionalInt groupSize,
        MlxOptionalInt bits,
        string mode,
        MlxArray globalScale,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_quantized_matmul", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int QuantizedMatMul
        (
        ref readonly MlxArray res,
        MlxArray x,
        MlxArray w,
        MlxArray scales,
        MlxArray biases,
        [MarshalAs(UnmanagedType.U1)] bool transpose,
        MlxOptionalInt groupSize,
        MlxOptionalInt bits,
        string mode,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_radians")]
    public static partial int Radians(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_real")]
    public static partial int Real(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_reciprocal")]
    public static partial int Reciprocal(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_remainder")]
    public static partial int Remainder
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_repeat_axis")]
    public static partial int RepeatAxis
        (
        ref readonly MlxArray res,
        MlxArray arr,
        int repeats,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_repeat")]
    public static partial int Repeat
        (
        ref readonly MlxArray res,
        MlxArray arr,
        int repeats,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_reshape")]
    public static partial int Reshape
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* shape,
        UIntPtr shapeNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_right_shift")]
    public static partial int RightShift
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_roll_axis")]
    public static partial int RollAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* shift,
        UIntPtr shiftNum,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_roll_axes")]
    public static partial int RollAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* shift,
        UIntPtr shiftNum,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_roll")]
    public static partial int Roll
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* shift,
        UIntPtr shiftNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_round")]
    public static partial int Round
        (
        ref readonly MlxArray res,
        MlxArray a,
        int decimals,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_rsqrt")]
    public static partial int RSqrt(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_scatter")]
    public static partial int Scatter
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxVectorArray indices,
        MlxArray updates,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_scatter_single")]
    public static partial int ScatterSingle
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray indices,
        MlxArray updates,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_scatter_add")]
    public static partial int ScatterAdd
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxVectorArray indices,
        MlxArray updates,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_scatter_add_single")]
    public static partial int ScatterAddSingle
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray indices,
        MlxArray updates,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_scatter_add_axis")]
    public static partial int ScatterAddAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray indices,
        MlxArray values,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_scatter_max")]
    public static partial int ScatterMax
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxVectorArray indices,
        MlxArray updates,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_scatter_max_single")]
    public static partial int ScatterMaxSingle
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray indices,
        MlxArray updates,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_scatter_min")]
    public static partial int ScatterMin
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxVectorArray indices,
        MlxArray updates,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_scatter_min_single")]
    public static partial int ScatterMinSingle
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray indices,
        MlxArray updates,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_scatter_prod")]
    public static partial int ScatterProd
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxVectorArray indices,
        MlxArray updates,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_scatter_prod_single")]
    public static partial int ScatterProdSingle
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray indices,
        MlxArray updates,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_segmented_mm")]
    public static partial int SegmentedMM
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxArray segments,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_sigmoid")]
    public static partial int Sigmoid(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_sign")]
    public static partial int Sign(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_sin")]
    public static partial int Sin(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_sinh")]
    public static partial int SinH(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_slice")]
    public static partial int Slice
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* start,
        UIntPtr startNum,
        int* stop,
        UIntPtr stopNum,
        int* strides,
        UIntPtr stridesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_slice_dynamic")]
    public static partial int SliceDynamic
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray start,
        int* axes,
        UIntPtr axesNum,
        int* sliceSize,
        UIntPtr sliceSizeNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_slice_update")]
    public static partial int SliceUpdate
        (
        ref readonly MlxArray res,
        MlxArray src,
        MlxArray update,
        int* start,
        UIntPtr startNum,
        int* stop,
        UIntPtr stopNum,
        int* strides,
        UIntPtr stridesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_slice_update_dynamic")]
    public static partial int SliceUpdateDynamic
        (
        ref readonly MlxArray res,
        MlxArray src,
        MlxArray update,
        MlxArray start,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_slice_update_add")]
    public static partial int SliceUpdateAdd
        (
        ref readonly MlxArray res,
        MlxArray src,
        MlxArray update,
        int* start,
        UIntPtr startNum,
        int* stop,
        UIntPtr stopNum,
        int* strides,
        UIntPtr stridesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_slice_update_max")]
    public static partial int SliceUpdateMax
        (
        ref readonly MlxArray res,
        MlxArray src,
        MlxArray update,
        int* start,
        UIntPtr startNum,
        int* stop,
        UIntPtr stopNum,
        int* strides,
        UIntPtr stridesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_slice_update_min")]
    public static partial int SliceUpdateMin
        (
        ref readonly MlxArray res,
        MlxArray src,
        MlxArray update,
        int* start,
        UIntPtr startNum,
        int* stop,
        UIntPtr stopNum,
        int* strides,
        UIntPtr stridesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_slice_update_prod")]
    public static partial int SliceUpdateProd
        (
        ref readonly MlxArray res,
        MlxArray src,
        MlxArray update,
        int* start,
        UIntPtr startNum,
        int* stop,
        UIntPtr stopNum,
        int* strides,
        UIntPtr stridesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_softmax_axes")]
    public static partial int SoftmaxAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool precise,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_softmax_axis")]
    public static partial int SoftmaxAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool precise,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_softmax")]
    public static partial int Softmax
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool precise,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_sort_axis")]
    public static partial int SortAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_sort")]
    public static partial int Sort(MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_split")]
    public static partial int Split
        (
        ref readonly MlxVectorArray res,
        MlxArray a,
        int numSplits,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_split_sections")]
    public static partial int SplitSections
        (
        ref readonly MlxVectorArray res,
        MlxArray a,
        int* indices,
        UIntPtr indicesNum,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_sqrt")]
    public static partial int Sqrt(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_square")]
    public static partial int Square(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_squeeze_axes")]
    public static partial int SqueezeAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_squeeze_axis")]
    public static partial int SqueezeAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_squeeze")]
    public static partial int Squeeze(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_stack_axis")]
    public static partial int StackAxis
        (
        ref readonly MlxArray res,
        MlxVectorArray arrays,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_stack")]
    public static partial int Stack
        (
        ref readonly MlxArray res,
        MlxVectorArray arrays,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_std_axes")]
    public static partial int StdAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        int ddof,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_std_axis")]
    public static partial int StdAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        int ddof,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_std")]
    public static partial int Std
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        int ddof,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_stop_gradient")]
    public static partial int StopGradient(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_subtract")]
    public static partial int Subtract
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_sum_axes")]
    public static partial int SumAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_sum_axis")]
    public static partial int SumAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_sum")]
    public static partial int Sum
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_swapaxes")]
    public static partial int SwapAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis1,
        int axis2,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_take_axis")]
    public static partial int TakeAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray indices,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_take")]
    public static partial int Take
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray indices,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_take_along_axis")]
    public static partial int TakeAlongAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray indices,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_tan")]
    public static partial int Tan(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_tanh")]
    public static partial int TanH(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_tensordot")]
    public static partial int TensorDot
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        int* axesA,
        UIntPtr axesANum,
        int* axesB,
        UIntPtr axesBNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_tensordot_axis")]
    public static partial int TensorDotAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        MlxArray b,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_tile")]
    public static partial int Tile
        (
        ref readonly MlxArray res,
        MlxArray arr,
        int* reps,
        UIntPtr repsNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_to_fp8")]
    public static partial int ToF8(ref readonly MlxArray res, MlxArray x, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_topk_axis")]
    public static partial int TopKAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int k,
        int axis,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_topk")]
    public static partial int TopK(ref readonly MlxArray res, MlxArray a, int k, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_trace")]
    public static partial int Trace
        (
        ref readonly MlxArray res,
        MlxArray a,
        int offset,
        int axis1,
        int axis2,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_transpose_axes")]
    public static partial int TransposeAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_transpose")]
    public static partial int Transpose(ref readonly MlxArray res, MlxArray a, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_tri")]
    public static partial int Tri
        (
        ref readonly MlxArray res,
        int n,
        int m,
        int k,
        DType type,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_tril")]
    public static partial int TriL(ref readonly MlxArray res, MlxArray x, int k, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_triu")]
    public static partial int TriU(ref readonly MlxArray res, MlxArray x, int k, MlxStream s);

    [LibraryImport("mlxc", EntryPoint = "mlx_unflatten")]
    public static partial int Unflatten
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        int* shape,
        UIntPtr shapeNum,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_var_axes")]
    public static partial int VarAxes
        (
        ref readonly MlxArray res,
        MlxArray a,
        int* axes,
        UIntPtr axesNum,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        int ddof,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_var_axis")]
    public static partial int VarAxis
        (
        ref readonly MlxArray res,
        MlxArray a,
        int axis,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        int ddof,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_var")]
    public static partial int Var
        (
        ref readonly MlxArray res,
        MlxArray a,
        [MarshalAs(UnmanagedType.U1)] bool keepdims,
        int ddof,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_view")]
    public static partial int View
        (
        ref readonly MlxArray res,
        MlxArray a,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_where")]
    public static partial int Where
        (
        ref readonly MlxArray res,
        MlxArray condition,
        MlxArray x,
        MlxArray y,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_zeros")]
    public static partial int Zeros
        (
        ref readonly MlxArray res,
        int* shape,
        UIntPtr shapeNum,
        DType dtype,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_zeros_like")]
    public static partial int ZerosLike(ref readonly MlxArray res, MlxArray a, MlxStream s);
}