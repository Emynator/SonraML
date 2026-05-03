using System.Collections.Concurrent;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.ExecutionManagement;

internal record class ReplyTo(ConcurrentQueue<MlxResponse> ResultQueue, EventWaitHandle Signal);

internal abstract record class MlxCommand(ReplyTo ReplyTo);

#region CreateOps

internal abstract record class CreationCommand(ReplyTo ReplyTo) : MlxCommand(ReplyTo);

internal record class CreateZeroOp(ReplyTo ReplyTo, DType Type, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateOneOp(ReplyTo ReplyTo, DType Type, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateScalarZeroOp(ReplyTo ReplyTo, DType Type) : CreationCommand(ReplyTo);

internal record class CreateScalarOneOp(ReplyTo ReplyTo, DType Type) : CreationCommand(ReplyTo);

internal record class CreateBoolScalarOp(ReplyTo ReplyTo, bool Scalar) : CreationCommand(ReplyTo);

internal record class CreateU8ScalarOp(ReplyTo ReplyTo, byte Scalar) : CreationCommand(ReplyTo);

internal record class CreateU16ScalarOp(ReplyTo ReplyTo, ushort Scalar) : CreationCommand(ReplyTo);

internal record class CreateU32ScalarOp(ReplyTo ReplyTo, uint Scalar) : CreationCommand(ReplyTo);

internal record class CreateU64ScalarOp(ReplyTo ReplyTo, ulong Scalar) : CreationCommand(ReplyTo);

internal record class CreateI8ScalarOp(ReplyTo ReplyTo, sbyte Scalar) : CreationCommand(ReplyTo);

internal record class CreateI16ScalarOp(ReplyTo ReplyTo, short Scalar) : CreationCommand(ReplyTo);

internal record class CreateI32ScalarOp(ReplyTo ReplyTo, int Scalar) : CreationCommand(ReplyTo);

internal record class CreateI64ScalarOp(ReplyTo ReplyTo, long Scalar) : CreationCommand(ReplyTo);

internal record class CreateF16ScalarOp(ReplyTo ReplyTo, Half Scalar) : CreationCommand(ReplyTo);

internal record class CreateF32ScalarOp(ReplyTo ReplyTo, float Scalar) : CreationCommand(ReplyTo);

internal record class CreateF64ScalarOp(ReplyTo ReplyTo, double Scalar) : CreationCommand(ReplyTo);

internal record class CreateBoolOp(ReplyTo ReplyTo, Memory<bool> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateU8Op(ReplyTo ReplyTo, Memory<byte> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateU16Op(ReplyTo ReplyTo, Memory<ushort> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateU32Op(ReplyTo ReplyTo, Memory<uint> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateU64Op(ReplyTo ReplyTo, Memory<ulong> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateI8Op(ReplyTo ReplyTo, Memory<sbyte> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateI16Op(ReplyTo ReplyTo, Memory<short> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateI32Op(ReplyTo ReplyTo, Memory<int> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateI64Op(ReplyTo ReplyTo, Memory<long> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateF16Op(ReplyTo ReplyTo, Memory<Half> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateF32Op(ReplyTo ReplyTo, Memory<float> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class CreateF64Op(ReplyTo ReplyTo, Memory<double> Array, TensorShape Shape) : CreationCommand(ReplyTo);

internal record class ArangeOp
    (
    ReplyTo ReplyTo,
    DType Type,
    double Start,
    double Stop,
    double Step
    ) : CreationCommand(ReplyTo);

internal record class LinspaceOp
    (
    ReplyTo ReplyTo,
    DType Type,
    double Start,
    double Stop,
    int Samples
    ) : CreationCommand(ReplyTo);

internal record class ConcatOp(ReplyTo ReplyTo, List<Guid> Tensors) : CreationCommand(ReplyTo);

internal record class ConcatAxisOp(ReplyTo ReplyTo, List<Guid> Tensors, int Axis) : CreationCommand(ReplyTo);

internal record class StackOp(ReplyTo ReplyTo, List<Guid> Tensors) : CreationCommand(ReplyTo);

internal record class StackAxisOp(ReplyTo ReplyTo, List<Guid> Tensors, int Axis) : CreationCommand(ReplyTo);

#endregion

#region DeleteOps

internal abstract record class DeleteCommand(ReplyTo ReplyTo) : MlxCommand(ReplyTo);

internal record class DeleteSingleOp(ReplyTo ReplyTo, Guid T) : DeleteCommand(ReplyTo);

internal record class DeleteManyOp(ReplyTo ReplyTo, List<Guid> Tensors) : DeleteCommand(ReplyTo);

#endregion

#region TensorOps

internal abstract record class TensorOpCommand(ReplyTo ReplyTo) : MlxCommand(ReplyTo);

internal abstract record class TensorArrayOpCommand(ReplyTo ReplyTo) : MlxCommand(ReplyTo);

internal record class GetShapeTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class IsScalarTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class EqualsOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class GetEnumeratorTensorOp(ReplyTo ReplyTo, Guid Tensor, DType Type) : TensorOpCommand(ReplyTo);

internal record class CopyTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class CopyFromTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class ConvertTensorOp(ReplyTo ReplyTo, Guid T, DType Type) : TensorOpCommand(ReplyTo);

internal record class ToStringOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class EnsureComputeTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

#region ArithmeticOps

internal record class AddTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class SubTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class MulTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class RecTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class DivTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class ModTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class RemTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class NegTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class AbsTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class SignTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

#endregion

#region LogicalOps

internal record class EqualTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class NotEqualTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class LessTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class LessEqualTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class GreaterTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class GreaterEqualTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class AndTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class OrTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class NotTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class IsNANTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class IsInfinityTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class IsFiniteTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class IsCloseTensorOp
    (
    ReplyTo ReplyTo,
    Guid A,
    Guid B,
    double RTol,
    double ATol,
    bool EqualNAN
    ) : TensorOpCommand(ReplyTo);

internal record class AllCloseTensorOp
    (
    ReplyTo ReplyTo,
    Guid A,
    Guid B,
    double RTol,
    double ATol,
    bool EqualNAN
    ) : TensorOpCommand(ReplyTo);

#endregion

#region BitwiseOps

internal record class BitwiseAndTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class BitwiseOrTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class BitwiseXorTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class BitwiseNotTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

#endregion

#region ExponentialOps

internal record class ExpTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class ExpM1TensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class LogTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class Log10TensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class Log2TensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class Log1PTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class SquareTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class SqrtTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class RSqrtTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class PowTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class LogSumExpTensorOp(ReplyTo ReplyTo, Guid T, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class LogSumExpAxisTensorOp
    (
    ReplyTo ReplyTo,
    Guid T,
    int Axis,
    bool KeepDims
    ) : TensorOpCommand(ReplyTo);

internal record class LogSumExpAxesTensorOp
    (
    ReplyTo ReplyTo,
    Guid T,
    int[] Axes,
    bool KeepDims
    ) : TensorOpCommand(ReplyTo);

internal record class LogAddExpTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

#endregion

#region TrigonometricOps

internal record class SinTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class SinHTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class ArcSinTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class ArcSinHTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class CosTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class CosHTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class ArcCosTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class ArcCosHTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class TanTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class TanHTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class ArcTanTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class ArcTanHTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class ArcTan2TensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

#endregion

#region Rounding

internal record class FloorTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class RoundTensorOp(ReplyTo ReplyTo, Guid T, int Decimals) : TensorOpCommand(ReplyTo);

internal record class CeilTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class ClipTensorOp(ReplyTo ReplyTo, Guid Tensor, Guid Min, Guid Max) : TensorOpCommand(ReplyTo);

internal record class FloorDivTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

#endregion

#region MatrixOps

internal record class MatMulTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class FmaTensorOp
    (
    ReplyTo ReplyTo,
    Guid A,
    Guid B,
    Guid C,
    float Alpha,
    float Beta
    ) : TensorOpCommand(ReplyTo);

internal record class TransposeTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class TransposeAxesTensorOp(ReplyTo ReplyTo, Guid T, int[] Axes) : TensorOpCommand(ReplyTo);

internal record class SwapAxesTensorOp(ReplyTo ReplyTo, Guid T, int A, int B) : TensorOpCommand(ReplyTo);

internal record class MoveAxisTensorOp(ReplyTo ReplyTo, Guid T, int Src, int Dest) : TensorOpCommand(ReplyTo);

internal record class DiagTensorOp(ReplyTo ReplyTo, Guid T, int Diagonal) : TensorOpCommand(ReplyTo);

#endregion

#region ShapeOps

internal record class ReshapeTensorOp(ReplyTo ReplyTo, Guid T, TensorShape Shape) : TensorOpCommand(ReplyTo);

internal record class FlattenTensorOp(ReplyTo ReplyTo, Guid T, int StartAxis, int EndAxis) : TensorOpCommand(ReplyTo);

internal record class ExpandDimsTensorOp(ReplyTo ReplyTo, Guid T, int Axis) : TensorOpCommand(ReplyTo);

internal record class ExpandDimsAxesTensorOp(ReplyTo ReplyTo, Guid T, int[] Axes) : TensorOpCommand(ReplyTo);

internal record class BroadcastToTensorOp(ReplyTo ReplyTo, Guid T, TensorShape Shape) : TensorOpCommand(ReplyTo);

#endregion

#region IndexingOps

internal record class SliceTensorOp
    (
    ReplyTo ReplyTo,
    Guid T,
    int[] Start,
    int[] Stop,
    int[] Strides
    ) : TensorOpCommand(ReplyTo);

internal record class DynamicSliceTensorOp
    (
    ReplyTo ReplyTo,
    Guid T,
    Guid Start,
    int[] Axes,
    int[] SliceSices
    ) : TensorOpCommand(ReplyTo);

internal record class SliceUpdateTensorOp
    (
    ReplyTo ReplyTo,
    Guid T,
    Guid Update,
    int[] Start,
    int[] Stop,
    int[] Strides
    ) : TensorOpCommand(ReplyTo);

internal record class TakeTensorOp(ReplyTo ReplyTo, Guid T, Guid Indices) : TensorOpCommand(ReplyTo);

internal record class TakeAxisTensorOp(ReplyTo ReplyTo, Guid T, Guid Indices, int Axis) : TensorOpCommand(ReplyTo);

internal record class TakeAlongAxisTensorOp(ReplyTo ReplyTo, Guid T, Guid Indices, int Axis) : TensorOpCommand(ReplyTo);

internal record class GatherTensorOp
    (
    ReplyTo ReplyTo,
    Guid T,
    List<Guid> Indices,
    int[] Axes,
    int[] SliceSices
    ) : TensorOpCommand(ReplyTo);

#endregion

#region SplitOps

internal record class SplitTensorOp(ReplyTo ReplyTo, Guid T, int NumSplits, int Axis) : TensorArrayOpCommand(ReplyTo);

internal record class SplitIndicesTensorOp
    (
    ReplyTo ReplyTo,
    Guid T,
    int[] Indices,
    int Axis
    ) : TensorArrayOpCommand(ReplyTo);

#endregion

#region PredicateOps

internal record class SumTensorOp(ReplyTo ReplyTo, Guid T, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class SumAxisTensorOp(ReplyTo ReplyTo, Guid T, int Axis, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class SumAxesTensorOp(ReplyTo ReplyTo, Guid T, int[] Axes, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class MinTensorOp(ReplyTo ReplyTo, Guid T, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class MinAxisTensorOp(ReplyTo ReplyTo, Guid T, int Axis, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class MinAxesTensorOp(ReplyTo ReplyTo, Guid T, int[] Axes, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class MaxTensorOp(ReplyTo ReplyTo, Guid T, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class MaxAxisTensorOp(ReplyTo ReplyTo, Guid T, int Axis, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class MaxAxesTensorOp(ReplyTo ReplyTo, Guid T, int[] Axes, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class MeanTensorOp(ReplyTo ReplyTo, Guid T, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class MeanAxisTensorOp(ReplyTo ReplyTo, Guid T, int Axis, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class MeanAxesTensorOp(ReplyTo ReplyTo, Guid T, int[] Axes, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class StdTensorOp(ReplyTo ReplyTo, Guid T, int Ddof, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class StdAxisTensorOp
    (
    ReplyTo ReplyTo,
    Guid T,
    int Axis,
    int Ddof,
    bool KeepDims
    ) : TensorOpCommand(ReplyTo);

internal record class StdAxesTensorOp
    (
    ReplyTo ReplyTo,
    Guid T,
    int[] Axes,
    int Ddof,
    bool KeepDims
    ) : TensorOpCommand(ReplyTo);

internal record class ArgMinTensorOp(ReplyTo ReplyTo, Guid T, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class ArgMinAxisTensorOp(ReplyTo ReplyTo, Guid T, int Axis, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class ArgMaxTensorOp(ReplyTo ReplyTo, Guid T, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class ArgMaxAxisTensorOp(ReplyTo ReplyTo, Guid T, int Axis, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class VarianceTensorOp(ReplyTo ReplyTo, Guid T, int Ddof, bool KeepDims) : TensorOpCommand(ReplyTo);

#endregion

#region SelectionOps

internal record class AllTensorOp(ReplyTo ReplyTo, Guid T, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class AllAxisTensorOp(ReplyTo ReplyTo, Guid T, int Axis, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class AllAxesTensorOp(ReplyTo ReplyTo, Guid T, int[] Axes, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class AnyTensorOp(ReplyTo ReplyTo, Guid T, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class AnyAxisTensorOp(ReplyTo ReplyTo, Guid T, int Axis, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class AnyAxesTensorOp(ReplyTo ReplyTo, Guid T, int[] Axes, bool KeepDims) : TensorOpCommand(ReplyTo);

internal record class WhereTensorOp(ReplyTo ReplyTo, Guid Cond, Guid IfTrue, Guid IfFalse) : TensorOpCommand(ReplyTo);

internal record class MinimumTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class MaximumTensorOp(ReplyTo ReplyTo, Guid A, Guid B) : TensorOpCommand(ReplyTo);

internal record class TopKTensorOp(ReplyTo ReplyTo, Guid T, int K) : TensorOpCommand(ReplyTo);

internal record class TopKAxisTensorOp(ReplyTo ReplyTo, Guid T, int K, int Axis) : TensorOpCommand(ReplyTo);

#endregion

#region LikeOps

internal record class ZerosLikeTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class OnesLikeTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

#endregion

#region NeuralOps

internal record class SigmoidTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class SoftmaxTensorOp(ReplyTo ReplyTo, Guid T, bool Precise) : TensorOpCommand(ReplyTo);

internal record class SoftmaxAxisTensorOp(ReplyTo ReplyTo, Guid T, int Axis, bool Precise) : TensorOpCommand(ReplyTo);

internal record class SoftmaxAxesTensorOp(ReplyTo ReplyTo, Guid T, int[] Axes, bool Precise) : TensorOpCommand(ReplyTo);

internal record class ErfTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

internal record class ErfInvTensorOp(ReplyTo ReplyTo, Guid T) : TensorOpCommand(ReplyTo);

#endregion

#endregion