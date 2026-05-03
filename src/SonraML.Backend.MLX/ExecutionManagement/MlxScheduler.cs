using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.Logging;
using SonraML.Backend.MLX.Extensions;
using SonraML.Backend.MLX.Implementations;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Core.Exceptions;
using SonraML.Core.Interfaces;

namespace SonraML.Backend.MLX.ExecutionManagement;

internal class MlxScheduler
{
    private readonly ILogger<MlxScheduler> logger;
    private readonly MlxTensorManager manager;
    private readonly ConcurrentQueue<MlxCommand> commandQueue;

    public MlxScheduler(ILogger<MlxScheduler> logger, MlxBackendGlobals globals, MlxTensorManager manager)
    {
        this.logger = logger;
        this.manager = manager;
        commandQueue = globals.CommandQueue;
    }

    public void Execute(CancellationToken ct)
    {
        manager.Init();
        
        while (!ct.IsCancellationRequested)
        {
            if (!commandQueue.TryDequeue(out var command))
            {
                Thread.Sleep(10);
                continue;
            }

            switch (command)
            {
                case CreationCommand creationCommand:
                    HandleCreationCommand(creationCommand);
                    break;

                case DeleteCommand deleteCommand:
                    HandleDeleteCommand(deleteCommand);
                    break;

                case TensorOpCommand tensorOpCommand:
                    HandleTensorOpCommand(tensorOpCommand);
                    break;

                case TensorArrayOpCommand tensorArrayOpCommand:
                    HandleTensorArrayOpCommand(tensorArrayOpCommand);
                    break;

                default:
                    logger.LogError("Received unknown command.");
                    break;
            }
        }
    }

    private void HandleCreationCommand(CreationCommand command)
    {
        switch (command)
        {
            case CreateZeroOp zeroOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateZeroOp));
                
                var zeroT = manager.Zero(zeroOp.Type, zeroOp.Shape);
                zeroOp.ReplyTo.SetTensorResponse(zeroT);
                break;

            case CreateOneOp oneOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateOneOp));
                
                var oneT = manager.One(oneOp.Type, oneOp.Shape);
                oneOp.ReplyTo.SetTensorResponse(oneT);
                break;

            case CreateScalarZeroOp scalarZeroOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateScalarZeroOp));
                
                var scalarZero = manager.ScalarZero(scalarZeroOp.Type);
                scalarZeroOp.ReplyTo.SetTensorResponse(scalarZero);
                break;

            case CreateScalarOneOp scalarOneOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateScalarOneOp));
                
                var scalarOne = manager.ScalarOne(scalarOneOp.Type);
                scalarOneOp.ReplyTo.SetTensorResponse(scalarOne);
                break;

            case CreateBoolScalarOp boolScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateBoolScalarOp));
                
                var boolScalar = manager.Create(boolScalarOp.Scalar);
                boolScalarOp.ReplyTo.SetTensorResponse(boolScalar);
                break;

            case CreateU8ScalarOp u8ScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateU8ScalarOp));
                
                var u8Scalar = manager.Create(u8ScalarOp.Scalar);
                u8ScalarOp.ReplyTo.SetTensorResponse(u8Scalar);
                break;

            case CreateU16ScalarOp u16ScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateU16ScalarOp));
                
                var u16Scalar = manager.Create(u16ScalarOp.Scalar);
                u16ScalarOp.ReplyTo.SetTensorResponse(u16Scalar);
                break;

            case CreateU32ScalarOp u32ScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateU32ScalarOp));
                
                var u32Scalar = manager.Create(u32ScalarOp.Scalar);
                u32ScalarOp.ReplyTo.SetTensorResponse(u32Scalar);
                break;

            case CreateU64ScalarOp u64ScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateU64ScalarOp));
                
                var u64Scalar = manager.Create(u64ScalarOp.Scalar);
                u64ScalarOp.ReplyTo.SetTensorResponse(u64Scalar);
                break;

            case CreateI8ScalarOp i8ScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateI8ScalarOp));
                
                var i8Scalar = manager.Create(i8ScalarOp.Scalar);
                i8ScalarOp.ReplyTo.SetTensorResponse(i8Scalar);
                break;

            case CreateI16ScalarOp i16ScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateI16ScalarOp));
                
                var i16Scalar = manager.Create(i16ScalarOp.Scalar);
                i16ScalarOp.ReplyTo.SetTensorResponse(i16Scalar);
                break;

            case CreateI32ScalarOp i32ScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateI32ScalarOp));
                
                var i32Scalar = manager.Create(i32ScalarOp.Scalar);
                i32ScalarOp.ReplyTo.SetTensorResponse(i32Scalar);
                break;

            case CreateI64ScalarOp i64ScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateI64ScalarOp));
                
                var i64Scalar = manager.Create(i64ScalarOp.Scalar);
                i64ScalarOp.ReplyTo.SetTensorResponse(i64Scalar);
                break;

            case CreateF16ScalarOp f16ScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateF16ScalarOp));
                
                var f16Scalar = manager.Create(f16ScalarOp.Scalar);
                f16ScalarOp.ReplyTo.SetTensorResponse(f16Scalar);
                break;

            case CreateF32ScalarOp f32ScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateF32ScalarOp));
                
                var f32Scalar = manager.Create(f32ScalarOp.Scalar);
                f32ScalarOp.ReplyTo.SetTensorResponse(f32Scalar);
                break;

            case CreateF64ScalarOp f64ScalarOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateF64ScalarOp));
                
                var f64Scalar = manager.Create(f64ScalarOp.Scalar);
                f64ScalarOp.ReplyTo.SetTensorResponse(f64Scalar);
                break;

            case CreateBoolOp boolOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateBoolOp));
                
                var arrayBool = manager.Create(boolOp.Array, boolOp.Shape);
                boolOp.ReplyTo.SetTensorResponse(arrayBool);
                break;

            case CreateU8Op u8Op:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateU8Op));
                
                var arrayU8 = manager.Create(u8Op.Array, u8Op.Shape);
                u8Op.ReplyTo.SetTensorResponse(arrayU8);
                break;

            case CreateU16Op u16Op:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateU16Op));
                
                var arrayU16 = manager.Create(u16Op.Array, u16Op.Shape);
                u16Op.ReplyTo.SetTensorResponse(arrayU16);
                break;

            case CreateU32Op u32Op:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateU32Op));
                
                var arrayU32 = manager.Create(u32Op.Array, u32Op.Shape);
                u32Op.ReplyTo.SetTensorResponse(arrayU32);
                break;

            case CreateU64Op u64Op:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateU64Op));
                
                var arrayU64 = manager.Create(u64Op.Array, u64Op.Shape);
                u64Op.ReplyTo.SetTensorResponse(arrayU64);
                break;

            case CreateI8Op i8Op:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateI8Op));
                
                var arrayI8 = manager.Create(i8Op.Array, i8Op.Shape);
                i8Op.ReplyTo.SetTensorResponse(arrayI8);
                break;

            case CreateI16Op i16Op:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateI16Op));
                
                var arrayI16 = manager.Create(i16Op.Array, i16Op.Shape);
                i16Op.ReplyTo.SetTensorResponse(arrayI16);
                break;

            case CreateI32Op i32Op:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateI32Op));
                
                var arrayI32 = manager.Create(i32Op.Array, i32Op.Shape);
                i32Op.ReplyTo.SetTensorResponse(arrayI32);
                break;

            case CreateI64Op i64Op:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateI64Op));
                
                var arrayI64 = manager.Create(i64Op.Array, i64Op.Shape);
                i64Op.ReplyTo.SetTensorResponse(arrayI64);
                break;

            case CreateF16Op f16Op:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateF16Op));
                
                var arrayF16 = manager.Create(f16Op.Array, f16Op.Shape);
                f16Op.ReplyTo.SetTensorResponse(arrayF16);
                break;

            case CreateF32Op f32Op:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateF32Op));
                
                var arrayF32 = manager.Create(f32Op.Array, f32Op.Shape);
                f32Op.ReplyTo.SetTensorResponse(arrayF32);
                break;

            case CreateF64Op f64Op:
                logger.LogTrace("Executing MLX OP {op}", nameof(CreateF64Op));
                
                var arrayF64 = manager.Create(f64Op.Array, f64Op.Shape);
                f64Op.ReplyTo.SetTensorResponse(arrayF64);
                break;

            case ArangeOp arangeOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArangeOp));
                
                var arange = manager.Arange(arangeOp.Type, arangeOp.Start, arangeOp.Stop, arangeOp.Step);
                arangeOp.ReplyTo.SetTensorResponse(arange);
                break;

            case LinspaceOp linspaceOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(LinspaceOp));
                
                var linspace = manager.Linspace(linspaceOp.Type, linspaceOp.Start, linspaceOp.Stop, linspaceOp.Samples);
                linspaceOp.ReplyTo.SetTensorResponse(linspace);
                break;

            case ConcatOp concatOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(ConcatOp));
                
                var toConcat = manager.Get(concatOp.Tensors);
                var concatResult = manager.Concat(toConcat);
                
                concatOp.ReplyTo.SetTensorResponse(concatResult);
                break;

            case ConcatAxisOp concatAxisOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(ConcatAxisOp));
                
                var toConcatAxis = manager.Get(concatAxisOp.Tensors);
                var concatAxisResult = manager.Concat(toConcatAxis, concatAxisOp.Axis);
                
                concatAxisOp.ReplyTo.SetTensorResponse(concatAxisResult);
                break;

            case StackOp stackOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(StackOp));
                
                var toStack = manager.Get(stackOp.Tensors);
                var stackResult = manager.Stack(toStack);
                
                stackOp.ReplyTo.SetTensorResponse(stackResult);
                break;

            case StackAxisOp stackAxisOp:
                logger.LogTrace("Executing MLX OP {op}", nameof(StackAxisOp));
                
                var toStackAxis = manager.Get(stackAxisOp.Tensors);
                var stackAxisResult = manager.Stack(toStackAxis, stackAxisOp.Axis);
                
                stackAxisOp.ReplyTo.SetTensorResponse(stackAxisResult);
                break;
        }
    }

    private void HandleDeleteCommand(DeleteCommand command)
    {
        switch (command)
        {
            case DeleteSingleOp single:
                logger.LogTrace("Executing MLX OP {op}", nameof(DeleteSingleOp));
                
                manager.Delete(single.T);
                single.ReplyTo.SetSuccessResponse();
                break;

            case DeleteManyOp many:
                logger.LogTrace("Executing MLX OP {op}", nameof(DeleteManyOp));
                
                manager.Delete(many.Tensors);
                many.ReplyTo.SetSuccessResponse();
                break;
        }
    }

    private void HandleTensorOpCommand(TensorOpCommand command)
    {
        switch (command)
        {
            case GetShapeTensorOp getShape:
                logger.LogTrace("Executing MLX OP {op}", nameof(GetShapeTensorOp));
                
                if (!GetTensor(getShape.ReplyTo, getShape.T, out var getShapeT))
                {
                    break;
                }
                
                getShape.ReplyTo.SetShapeResponse(getShapeT.Shape);
                break;

            case IsScalarTensorOp isScalar:
                logger.LogTrace("Executing MLX OP {op}", nameof(IsScalarTensorOp));
                
                if (!GetTensor(isScalar.ReplyTo, isScalar.T, out var isScalarT))
                {
                    break;
                }
                
                isScalar.ReplyTo.SetIsScalarResponse(isScalarT.IsScalar);
                break;
            
            case EqualsOp equals:
                logger.LogTrace("Executing MLX OP {op}", nameof(EqualsOp));
                
                if (!GetTensor(equals.ReplyTo, equals.A, out var equalsA))
                {
                    break;
                }
                if (!GetTensor(equals.ReplyTo, equals.B, out var equalsB))
                {
                    break;
                }
                
                equals.ReplyTo.SetEqualsResponse(Equals(equalsB));
                break;

            case GetEnumeratorTensorOp getEnumerator:
                logger.LogTrace("Executing MLX OP {op}", nameof(GetEnumeratorTensorOp));
                
                if (!GetTensor(getEnumerator.ReplyTo, getEnumerator.Tensor, out var getEnumeratorT))
                {
                    break;
                }

                object? enumeratorResult = getEnumerator.Type switch
                {
                    DType.Bool => getEnumeratorT.GetEnumerator<bool>(),
                    DType.UInt8 => getEnumeratorT.GetEnumerator<byte>(),
                    DType.UInt16 => getEnumeratorT.GetEnumerator<ushort>(),
                    DType.UInt32 => getEnumeratorT.GetEnumerator<uint>(),
                    DType.UInt64 => getEnumeratorT.GetEnumerator<ulong>(),
                    DType.Int8 => getEnumeratorT.GetEnumerator<sbyte>(),
                    DType.Int16 => getEnumeratorT.GetEnumerator<short>(),
                    DType.Int32 => getEnumeratorT.GetEnumerator<int>(),
                    DType.Int64 => getEnumeratorT.GetEnumerator<long>(),
                    DType.Float16 => getEnumeratorT.GetEnumerator<Half>(),
                    DType.Float32 => getEnumeratorT.GetEnumerator<float>(),
                    DType.Float64 => getEnumeratorT.GetEnumerator<double>(),
                    _ => null,
                };
                if (enumeratorResult is null)
                {
                    getEnumerator.ReplyTo.SetErrorResponse("Failed to get Enumerator.");
                    break;
                }
                
                getEnumerator.ReplyTo.SetEnumeratorResponse(enumeratorResult);
                break;

            case CopyTensorOp copy:
                logger.LogTrace("Executing MLX OP {op}", nameof(CopyTensorOp));
                
                if (!GetTensor(copy.ReplyTo, copy.T, out var copyT))
                {
                    break;
                }
                
                var copyResult = copyT.Copy();
                copy.ReplyTo.SetTensorResponse(copyResult);
                break;

            case CopyFromTensorOp copyFrom:
                logger.LogTrace("Executing MLX OP {op}", nameof(CopyFromTensorOp));
                
                if (!GetTensor(copyFrom.ReplyTo, copyFrom.A, out var copyFromA))
                {
                    break;
                }
                if (!GetTensor(copyFrom.ReplyTo, copyFrom.B, out var copyFromB))
                {
                    break;
                }
                
                copyFromA.CopyFrom(copyFromB);
                copyFrom.ReplyTo.SetSuccessResponse();
                break;
            
            case ConvertTensorOp convert:
                logger.LogTrace("Executing MLX OP {op}", nameof(ConvertTensorOp));
                
                if (!GetTensor(convert.ReplyTo, convert.T, out var convertT))
                {
                    break;
                }
                
                convert.ReplyTo.SetTensorResponse(convertT.ConvertTo(convert.Type));
                break;
            
            case ToStringOp toString:
                logger.LogTrace("Executing MLX OP {op}", nameof(ToStringOp));
                
                if (!GetTensor(toString.ReplyTo, toString.T, out var toStringT))
                {
                    break;
                }
                
                toString.ReplyTo.SetToStringResponse(toStringT.ToString());
                break;

            case EnsureComputeTensorOp ensureCompute:
                logger.LogTrace("Executing MLX OP {op}", nameof(EnsureComputeTensorOp));
                
                if (!GetTensor(ensureCompute.ReplyTo, ensureCompute.T, out var ensureComputeT))
                {
                    break;
                }
                
                ensureComputeT.EnsureCompute();
                ensureCompute.ReplyTo.SetSuccessResponse();
                break;

            case AddTensorOp add:
                logger.LogTrace("Executing MLX OP {op}", nameof(AddTensorOp));
                
                if (!GetTensor(add.ReplyTo, add.A, out var addA))
                {
                    break;
                }
                if (!GetTensor(add.ReplyTo, add.B, out var addB))
                {
                    break;
                }
                
                var addResult = addA.Add(addB);
                add.ReplyTo.SetTensorResponse(addResult);
                break;

            case SubTensorOp sub:
                logger.LogTrace("Executing MLX OP {op}", nameof(SubTensorOp));
                
                if (!GetTensor(sub.ReplyTo, sub.A, out var subA))
                {
                    break;
                }
                if (!GetTensor(sub.ReplyTo, sub.B, out var subB))
                {
                    break;
                }
                
                var subResult = subA.Sub(subB);
                sub.ReplyTo.SetTensorResponse(subResult);
                break;

            case MulTensorOp mul:
                logger.LogTrace("Executing MLX OP {op}", nameof(MulTensorOp));
                
                if (!GetTensor(mul.ReplyTo, mul.A, out var mulA))
                {
                    break;
                }
                if (!GetTensor(mul.ReplyTo, mul.B, out var mulB))
                {
                    break;
                }
                
                var mulResult = mulA.Mul(mulB);
                mul.ReplyTo.SetTensorResponse(mulResult);
                break;

            case RecTensorOp rec:
                logger.LogTrace("Executing MLX OP {op}", nameof(RecTensorOp));
                
                if (!GetTensor(rec.ReplyTo, rec.T, out var recT))
                {
                    break;
                }

                var recResult = recT.Rec();
                rec.ReplyTo.SetTensorResponse(recResult);
                break;

            case DivTensorOp div:
                logger.LogTrace("Executing MLX OP {op}", nameof(DivTensorOp));
                
                if (!GetTensor(div.ReplyTo, div.A, out var divA))
                {
                    break;
                }
                if (!GetTensor(div.ReplyTo, div.B, out var divB))
                {
                    break;
                }
                
                var divResult = divA.Div(divB);
                div.ReplyTo.SetTensorResponse(divResult);
                break;

            case ModTensorOp mod:
                logger.LogTrace("Executing MLX OP {op}", nameof(ModTensorOp));
                
                if (!GetTensor(mod.ReplyTo, mod.A, out var modA))
                {
                    break;
                }
                if (!GetTensor(mod.ReplyTo, mod.B, out var modB))
                {
                    break;
                }
                
                var modResult = modA.Mod(modB);
                mod.ReplyTo.SetTensorResponse(modResult);
                break;

            case RemTensorOp rem:
                logger.LogTrace("Executing MLX OP {op}", nameof(RemTensorOp));
                
                if (!GetTensor(rem.ReplyTo, rem.A, out var remA))
                {
                    break;
                }
                if (!GetTensor(rem.ReplyTo, rem.B, out var remB))
                {
                    break;
                }
                
                var remResult = remA.Rem(remB);
                rem.ReplyTo.SetTensorResponse(remResult);
                break;

            case NegTensorOp neg:
                logger.LogTrace("Executing MLX OP {op}", nameof(NegTensorOp));
                
                if (!GetTensor(neg.ReplyTo, neg.T, out var negT))
                {
                    break;
                }

                var negResult = negT.Neg();
                neg.ReplyTo.SetTensorResponse(negResult);
                break;

            case AbsTensorOp abs:
                logger.LogTrace("Executing MLX OP {op}", nameof(AbsTensorOp));
                
                if (!GetTensor(abs.ReplyTo, abs.T, out var absT))
                {
                    break;
                }

                var absResult = absT.Abs();
                abs.ReplyTo.SetTensorResponse(absResult);
                break;

            case SignTensorOp sign:
                logger.LogTrace("Executing MLX OP {op}", nameof(SignTensorOp));
                
                if (!GetTensor(sign.ReplyTo, sign.T, out var signT))
                {
                    break;
                }

                var signResult = signT.Sign();
                sign.ReplyTo.SetTensorResponse(signResult);
                break;

            case EqualTensorOp equal:
                logger.LogTrace("Executing MLX OP {op}", nameof(EqualTensorOp));
                
                if (!GetTensor(equal.ReplyTo, equal.A, out var equalA))
                {
                    break;
                }
                if (!GetTensor(equal.ReplyTo, equal.B, out var equalB))
                {
                    break;
                }
                
                var equalResult = equalA.Equal(equalB);
                equal.ReplyTo.SetTensorResponse(equalResult);
                break;

            case NotEqualTensorOp notEqual:
                logger.LogTrace("Executing MLX OP {op}", nameof(NotEqualTensorOp));
                
                if (!GetTensor(notEqual.ReplyTo, notEqual.A, out var notEqualA))
                {
                    break;
                }
                if (!GetTensor(notEqual.ReplyTo, notEqual.B, out var notEqualB))
                {
                    break;
                }
                
                var notEqualResult = notEqualA.NotEqual(notEqualB);
                notEqual.ReplyTo.SetTensorResponse(notEqualResult);
                break;

            case LessTensorOp less:
                logger.LogTrace("Executing MLX OP {op}", nameof(LessTensorOp));
                
                if (!GetTensor(less.ReplyTo, less.A, out var lessA))
                {
                    break;
                }
                if (!GetTensor(less.ReplyTo, less.B, out var lessB))
                {
                    break;
                }
                
                var lessResult = lessA.Less(lessB);
                less.ReplyTo.SetTensorResponse(lessResult);
                break;

            case LessEqualTensorOp lessEqual:
                logger.LogTrace("Executing MLX OP {op}", nameof(LessEqualTensorOp));
                
                if (!GetTensor(lessEqual.ReplyTo, lessEqual.A, out var lessEqualA))
                {
                    break;
                }
                if (!GetTensor(lessEqual.ReplyTo, lessEqual.B, out var lessEqualB))
                {
                    break;
                }
                
                var lessEqualResult = lessEqualA.LessEqual(lessEqualB);
                lessEqual.ReplyTo.SetTensorResponse(lessEqualResult);
                break;

            case GreaterTensorOp greater:
                logger.LogTrace("Executing MLX OP {op}", nameof(GreaterTensorOp));
                
                if (!GetTensor(greater.ReplyTo, greater.A, out var greaterA))
                {
                    break;
                }
                if (!GetTensor(greater.ReplyTo, greater.B, out var greaterB))
                {
                    break;
                }
                
                var greaterResult = greaterA.Greater(greaterB);
                greater.ReplyTo.SetTensorResponse(greaterResult);
                break;

            case GreaterEqualTensorOp greaterEqual:
                logger.LogTrace("Executing MLX OP {op}", nameof(GreaterEqualTensorOp));
                
                if (!GetTensor(greaterEqual.ReplyTo, greaterEqual.A, out var greaterEqualA))
                {
                    break;
                }
                if (!GetTensor(greaterEqual.ReplyTo, greaterEqual.B, out var greaterEqualB))
                {
                    break;
                }
                
                var greaterEqualResult = greaterEqualA.GreaterEqual(greaterEqualB);
                greaterEqual.ReplyTo.SetTensorResponse(greaterEqualResult);
                break;

            case AndTensorOp and:
                logger.LogTrace("Executing MLX OP {op}", nameof(AndTensorOp));
                
                if (!GetTensor(and.ReplyTo, and.A, out var andA))
                {
                    break;
                }
                if (!GetTensor(and.ReplyTo, and.B, out var andB))
                {
                    break;
                }
                
                var andResult = andA.And(andB);
                and.ReplyTo.SetTensorResponse(andResult);
                break;

            case OrTensorOp or:
                logger.LogTrace("Executing MLX OP {op}", nameof(OrTensorOp));
                
                if (!GetTensor(or.ReplyTo, or.A, out var orA))
                {
                    break;
                }
                if (!GetTensor(or.ReplyTo, or.B, out var orB))
                {
                    break;
                }
                
                var orResult = orA.Or(orB);
                or.ReplyTo.SetTensorResponse(orResult);
                break;

            case NotTensorOp not:
                logger.LogTrace("Executing MLX OP {op}", nameof(NotTensorOp));
                
                if (!GetTensor(not.ReplyTo, not.T, out var notT))
                {
                    break;
                }

                var notResult = notT.Not();
                not.ReplyTo.SetTensorResponse(notResult);
                break;

            case IsNANTensorOp isNAN:
                logger.LogTrace("Executing MLX OP {op}", nameof(IsNANTensorOp));
                
                if (!GetTensor(isNAN.ReplyTo, isNAN.T, out var isNANT))
                {
                    break;
                }

                var isNANResult = isNANT.IsNAN();
                isNAN.ReplyTo.SetTensorResponse(isNANResult);
                break;

            case IsInfinityTensorOp isInfinity:
                logger.LogTrace("Executing MLX OP {op}", nameof(IsInfinityTensorOp));
                
                if (!GetTensor(isInfinity.ReplyTo, isInfinity.T, out var isInfinityT))
                {
                    break;
                }

                var isInfinityResult = isInfinityT.IsInfinity();
                isInfinity.ReplyTo.SetTensorResponse(isInfinityResult);
                break;

            case IsFiniteTensorOp isFinite:
                logger.LogTrace("Executing MLX OP {op}", nameof(IsFiniteTensorOp));
                
                if (!GetTensor(isFinite.ReplyTo, isFinite.T, out var isFiniteT))
                {
                    break;
                }

                var isFiniteResult = isFiniteT.IsFinite();
                isFinite.ReplyTo.SetTensorResponse(isFiniteResult);
                break;

            case IsCloseTensorOp isClose:
                logger.LogTrace("Executing MLX OP {op}", nameof(IsCloseTensorOp));
                
                if (!GetTensor(isClose.ReplyTo, isClose.A, out var isCloseA))
                {
                    break;
                }
                if (!GetTensor(isClose.ReplyTo, isClose.B, out var isCloseB))
                {
                    break;
                }

                var isCloseResult = isCloseA.IsClose(isCloseB, isClose.RTol, isClose.ATol, isClose.EqualNAN);
                isClose.ReplyTo.SetTensorResponse(isCloseResult);
                break;

            case AllCloseTensorOp allClose:
                logger.LogTrace("Executing MLX OP {op}", nameof(AllCloseTensorOp));
                
                if (!GetTensor(allClose.ReplyTo, allClose.A, out var allCloseA))
                {
                    break;
                }
                if (!GetTensor(allClose.ReplyTo, allClose.B, out var allCloseB))
                {
                    break;
                }

                var allCloseResult = allCloseA.AllClose(allCloseB, allClose.RTol, allClose.ATol, allClose.EqualNAN);
                allClose.ReplyTo.SetTensorResponse(allCloseResult);
                break;

            case BitwiseAndTensorOp bitwiseAnd:
                logger.LogTrace("Executing MLX OP {op}", nameof(BitwiseAndTensorOp));
                
                if (!GetTensor(bitwiseAnd.ReplyTo, bitwiseAnd.A, out var bitwiseAndA))
                {
                    break;
                }
                if (!GetTensor(bitwiseAnd.ReplyTo, bitwiseAnd.B, out var bitwiseAndB))
                {
                    break;
                }
                
                var bitwiseAndResult = bitwiseAndA.BitwiseAnd(bitwiseAndB);
                bitwiseAnd.ReplyTo.SetTensorResponse(bitwiseAndResult);
                break;

            case BitwiseOrTensorOp bitwiseOr:
                logger.LogTrace("Executing MLX OP {op}", nameof(BitwiseOrTensorOp));
                
                if (!GetTensor(bitwiseOr.ReplyTo, bitwiseOr.A, out var bitwiseOrA))
                {
                    break;
                }
                if (!GetTensor(bitwiseOr.ReplyTo, bitwiseOr.B, out var bitwiseOrB))
                {
                    break;
                }
                
                var bitwiseOrResult = bitwiseOrA.BitwiseOr(bitwiseOrB);
                bitwiseOr.ReplyTo.SetTensorResponse(bitwiseOrResult);
                break;

            case BitwiseXorTensorOp bitwiseXor:
                logger.LogTrace("Executing MLX OP {op}", nameof(BitwiseXorTensorOp));
                
                if (!GetTensor(bitwiseXor.ReplyTo, bitwiseXor.A, out var bitwiseXorA))
                {
                    break;
                }
                if (!GetTensor(bitwiseXor.ReplyTo, bitwiseXor.B, out var bitwiseXorB))
                {
                    break;
                }
                
                var bitwiseXorResult = bitwiseXorA.BitwiseXor(bitwiseXorB);
                bitwiseXor.ReplyTo.SetTensorResponse(bitwiseXorResult);
                break;

            case BitwiseNotTensorOp bitwiseNot:
                logger.LogTrace("Executing MLX OP {op}", nameof(BitwiseNotTensorOp));
                
                if (!GetTensor(bitwiseNot.ReplyTo, bitwiseNot.T, out var bitwiseNotT))
                {
                    break;
                }

                var bitwiseNotResult = bitwiseNotT.BitwiseNot();
                bitwiseNot.ReplyTo.SetTensorResponse(bitwiseNotResult);
                break;

            case ExpTensorOp exp:
                logger.LogTrace("Executing MLX OP {op}", nameof(ExpTensorOp));
                
                if (!GetTensor(exp.ReplyTo, exp.T, out var expT))
                {
                    break;
                }

                var expResult = expT.Exp();
                exp.ReplyTo.SetTensorResponse(expResult);
                break;

            case ExpM1TensorOp expM1:
                logger.LogTrace("Executing MLX OP {op}", nameof(ExpM1TensorOp));
                
                if (!GetTensor(expM1.ReplyTo, expM1.T, out var expM1T))
                {
                    break;
                }

                var expM1Result = expM1T.ExpM1();
                expM1.ReplyTo.SetTensorResponse(expM1Result);
                break;

            case LogTensorOp log:
                logger.LogTrace("Executing MLX OP {op}", nameof(LogTensorOp));
                
                if (!GetTensor(log.ReplyTo, log.T, out var logT))
                {
                    break;
                }

                var logResult = logT.Log();
                log.ReplyTo.SetTensorResponse(logResult);
                break;

            case Log10TensorOp log10:
                logger.LogTrace("Executing MLX OP {op}", nameof(Log10TensorOp));
                
                if (!GetTensor(log10.ReplyTo, log10.T, out var log10T))
                {
                    break;
                }

                var log10Result = log10T.Log10();
                log10.ReplyTo.SetTensorResponse(log10Result);
                break;

            case Log2TensorOp log2:
                logger.LogTrace("Executing MLX OP {op}", nameof(Log2TensorOp));
                
                if (!GetTensor(log2.ReplyTo, log2.T, out var log2T))
                {
                    break;
                }

                var log2Result = log2T.Log2();
                log2.ReplyTo.SetTensorResponse(log2Result);
                break;

            case Log1PTensorOp log1P:
                logger.LogTrace("Executing MLX OP {op}", nameof(Log1PTensorOp));
                
                if (!GetTensor(log1P.ReplyTo, log1P.T, out var log1PT))
                {
                    break;
                }

                var log1PResult = log1PT.Log1P();
                log1P.ReplyTo.SetTensorResponse(log1PResult);
                break;

            case SquareTensorOp square:
                logger.LogTrace("Executing MLX OP {op}", nameof(SquareTensorOp));
                
                if (!GetTensor(square.ReplyTo, square.T, out var squareT))
                {
                    break;
                }

                var squareResult = squareT.Square();
                square.ReplyTo.SetTensorResponse(squareResult);
                break;

            case SqrtTensorOp sqrt:
                logger.LogTrace("Executing MLX OP {op}", nameof(SqrtTensorOp));
                
                if (!GetTensor(sqrt.ReplyTo, sqrt.T, out var sqrtT))
                {
                    break;
                }

                var sqrtResult = sqrtT.Sqrt();
                sqrt.ReplyTo.SetTensorResponse(sqrtResult);
                break;

            case RSqrtTensorOp rSqrt:
                logger.LogTrace("Executing MLX OP {op}", nameof(RSqrtTensorOp));
                
                if (!GetTensor(rSqrt.ReplyTo, rSqrt.T, out var rsqrtT))
                {
                    break;
                }

                var rsqrtResult = rsqrtT.RSqrt();
                rSqrt.ReplyTo.SetTensorResponse(rsqrtResult);
                break;

            case PowTensorOp pow:
                logger.LogTrace("Executing MLX OP {op}", nameof(PowTensorOp));
                
                if (!GetTensor(pow.ReplyTo, pow.A, out var powA))
                {
                    break;
                }
                if (!GetTensor(pow.ReplyTo, pow.B, out var powB))
                {
                    break;
                }
                
                var powResult = powA.Pow(powB);
                pow.ReplyTo.SetTensorResponse(powResult);
                break;

            case LogSumExpTensorOp logSumExp:
                logger.LogTrace("Executing MLX OP {op}", nameof(LogSumExpTensorOp));
                
                if (!GetTensor(logSumExp.ReplyTo, logSumExp.T, out var logSumExpT))
                {
                    break;
                }

                var logSumExpResult = logSumExpT.LogSumExp(logSumExp.KeepDims);
                logSumExp.ReplyTo.SetTensorResponse(logSumExpResult);
                break;

            case LogSumExpAxisTensorOp logSumExpAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(LogSumExpAxisTensorOp));
                
                if (!GetTensor(logSumExpAxis.ReplyTo, logSumExpAxis.T, out var logSumExpAxisT))
                {
                    break;
                }

                var logSumExpAxisResult = logSumExpAxisT.LogSumExp(logSumExpAxis.Axis, logSumExpAxis.KeepDims);
                logSumExpAxis.ReplyTo.SetTensorResponse(logSumExpAxisResult);
                break;

            case LogSumExpAxesTensorOp logSumExpAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(LogSumExpAxesTensorOp));
                
                if (!GetTensor(logSumExpAxes.ReplyTo, logSumExpAxes.T, out var logSumExpAxesT))
                {
                    break;
                }

                var logSumExpAxesResult = logSumExpAxesT.LogSumExp(logSumExpAxes.Axes, logSumExpAxes.KeepDims);
                logSumExpAxes.ReplyTo.SetTensorResponse(logSumExpAxesResult);
                break;

            case LogAddExpTensorOp logAddExp:
                logger.LogTrace("Executing MLX OP {op}", nameof(LogAddExpTensorOp));
                
                if (!GetTensor(logAddExp.ReplyTo, logAddExp.A, out var logAddExpA))
                {
                    break;
                }
                if (!GetTensor(logAddExp.ReplyTo, logAddExp.B, out var logAddExpB))
                {
                    break;
                }
                
                var logAddExpResult = logAddExpA.LogAddExp(logAddExpB);
                logAddExp.ReplyTo.SetTensorResponse(logAddExpResult);
                break;

            case SinTensorOp sin:
                logger.LogTrace("Executing MLX OP {op}", nameof(SinTensorOp));
                
                if (!GetTensor(sin.ReplyTo, sin.T, out var sinT))
                {
                    break;
                }

                var sinResult = sinT.Sin();
                sin.ReplyTo.SetTensorResponse(sinResult);
                break;

            case SinHTensorOp sinH:
                logger.LogTrace("Executing MLX OP {op}", nameof(SinHTensorOp));
                
                if (!GetTensor(sinH.ReplyTo, sinH.T, out var sinHT))
                {
                    break;
                }

                var sinHResult = sinHT.SinH();
                sinH.ReplyTo.SetTensorResponse(sinHResult);
                break;

            case ArcSinTensorOp arcSin:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArcSinTensorOp));
                
                if (!GetTensor(arcSin.ReplyTo, arcSin.T, out var arcSinT))
                {
                    break;
                }

                var arcSinResult = arcSinT.ArcSin();
                arcSin.ReplyTo.SetTensorResponse(arcSinResult);
                break;

            case ArcSinHTensorOp arcSinH:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArcSinHTensorOp));
                
                if (!GetTensor(arcSinH.ReplyTo, arcSinH.T, out var arcSinHT))
                {
                    break;
                }

                var arcSinHResult = arcSinHT.ArcSinH();
                arcSinH.ReplyTo.SetTensorResponse(arcSinHResult);
                break;

            case CosTensorOp cos:
                logger.LogTrace("Executing MLX OP {op}", nameof(CosTensorOp));
                
                if (!GetTensor(cos.ReplyTo, cos.T, out var cosT))
                {
                    break;
                }

                var cosResult = cosT.Cos();
                cos.ReplyTo.SetTensorResponse(cosResult);
                break;

            case CosHTensorOp cosH:
                logger.LogTrace("Executing MLX OP {op}", nameof(CosHTensorOp));
                
                if (!GetTensor(cosH.ReplyTo, cosH.T, out var cosHT))
                {
                    break;
                }

                var cosHResult = cosHT.CosH();
                cosH.ReplyTo.SetTensorResponse(cosHResult);
                break;

            case ArcCosTensorOp arcCos:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArcCosTensorOp));
                
                if (!GetTensor(arcCos.ReplyTo, arcCos.T, out var arcCosT))
                {
                    break;
                }

                var arcCosResult = arcCosT.ArcCos();
                arcCos.ReplyTo.SetTensorResponse(arcCosResult);
                break;

            case ArcCosHTensorOp arcCosH:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArcCosHTensorOp));
                
                if (!GetTensor(arcCosH.ReplyTo, arcCosH.T, out var arcCosHT))
                {
                    break;
                }

                var arcCosHResult = arcCosHT.ArcCosH();
                arcCosH.ReplyTo.SetTensorResponse(arcCosHResult);
                break;

            case TanTensorOp tan:
                logger.LogTrace("Executing MLX OP {op}", nameof(TanTensorOp));
                
                if (!GetTensor(tan.ReplyTo, tan.T, out var tanT))
                {
                    break;
                }

                var tanResult = tanT.Tan();
                tan.ReplyTo.SetTensorResponse(tanResult);
                break;

            case TanHTensorOp tanH:
                logger.LogTrace("Executing MLX OP {op}", nameof(TanHTensorOp));
                
                if (!GetTensor(tanH.ReplyTo, tanH.T, out var tanHT))
                {
                    break;
                }

                var tanHResult = tanHT.TanH();
                tanH.ReplyTo.SetTensorResponse(tanHResult);
                break;

            case ArcTanTensorOp arcTan:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArcTanTensorOp));
                
                if (!GetTensor(arcTan.ReplyTo, arcTan.T, out var arcTanT))
                {
                    break;
                }

                var arcTanResult = arcTanT.ArcTan();
                arcTan.ReplyTo.SetTensorResponse(arcTanResult);
                break;

            case ArcTanHTensorOp arcTanH:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArcTanHTensorOp));
                
                if (!GetTensor(arcTanH.ReplyTo, arcTanH.T, out var arcTanHT))
                {
                    break;
                }

                var arcTanHResult = arcTanHT.ArcTanH();
                arcTanH.ReplyTo.SetTensorResponse(arcTanHResult);
                break;

            case ArcTan2TensorOp arcTan2:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArcTan2TensorOp));
                
                if (!GetTensor(arcTan2.ReplyTo, arcTan2.A, out var arcTan2A))
                {
                    break;
                }
                if (!GetTensor(arcTan2.ReplyTo, arcTan2.B, out var arcTan2B))
                {
                    break;
                }
                
                var arcTan2Result = arcTan2A.ArcTan2(arcTan2B);
                arcTan2.ReplyTo.SetTensorResponse(arcTan2Result);
                break;

            case FloorTensorOp floor:
                logger.LogTrace("Executing MLX OP {op}", nameof(FloorTensorOp));
                
                if (!GetTensor(floor.ReplyTo, floor.T, out var floorT))
                {
                    break;
                }

                var floorResult = floorT.Floor();
                floor.ReplyTo.SetTensorResponse(floorResult);
                break;

            case RoundTensorOp round:
                logger.LogTrace("Executing MLX OP {op}", nameof(RoundTensorOp));
                
                if (!GetTensor(round.ReplyTo, round.T, out var roundT))
                {
                    break;
                }

                var roundResult = roundT.Round(round.Decimals);
                round.ReplyTo.SetTensorResponse(roundResult);
                break;

            case CeilTensorOp ceil:
                logger.LogTrace("Executing MLX OP {op}", nameof(CeilTensorOp));
                
                if (!GetTensor(ceil.ReplyTo, ceil.T, out var ceilT))
                {
                    break;
                }

                var ceilResult = ceilT.Ceil();
                ceil.ReplyTo.SetTensorResponse(ceilResult);
                break;
            
            case ClipTensorOp clip:
                logger.LogTrace("Executing MLX OP {op}", nameof(ClipTensorOp));
                
                if (!GetTensor(clip.ReplyTo, clip.Tensor, out var clipT))
                {
                    break;
                }
                if (!GetTensor(clip.ReplyTo, clip.Min, out var clipMin))
                {
                    break;
                }
                if (!GetTensor(clip.ReplyTo, clip.Max, out var clipMax))
                {
                    break;
                }

                var clipResult = clipT.Clip(clipMin, clipMax);
                clip.ReplyTo.SetTensorResponse(clipResult);
                break;

            case FloorDivTensorOp floorDiv:
                logger.LogTrace("Executing MLX OP {op}", nameof(FloorDivTensorOp));
                
                if (!GetTensor(floorDiv.ReplyTo, floorDiv.A, out var floorDivA))
                {
                    break;
                }
                if (!GetTensor(floorDiv.ReplyTo, floorDiv.B, out var floorDivB))
                {
                    break;
                }
                
                var floorDivResult = floorDivA.FloorDiv(floorDivB);
                floorDiv.ReplyTo.SetTensorResponse(floorDivResult);
                break;

            case MatMulTensorOp matMul:
                logger.LogTrace("Executing MLX OP {op}", nameof(MatMulTensorOp));
                
                if (!GetTensor(matMul.ReplyTo, matMul.A, out var matMulA))
                {
                    break;
                }
                if (!GetTensor(matMul.ReplyTo, matMul.B, out var matMulB))
                {
                    break;
                }
                
                var matMulResult = matMulA.MatMul(matMulB);
                matMul.ReplyTo.SetTensorResponse(matMulResult);
                break;

            case FmaTensorOp fma:
                logger.LogTrace("Executing MLX OP {op}", nameof(FmaTensorOp));
                
                if (!GetTensor(fma.ReplyTo, fma.A, out var fmaA))
                {
                    break;
                }
                if (!GetTensor(fma.ReplyTo, fma.B, out var fmaB))
                {
                    break;
                }
                if (!GetTensor(fma.ReplyTo, fma.C, out var fmaC))
                {
                    break;
                }
                
                var fmaResult = fmaA.Fma(fmaB, fmaC, fma.Alpha, fma.Beta);
                fma.ReplyTo.SetTensorResponse(fmaResult);
                break;

            case TransposeTensorOp transpose:
                logger.LogTrace("Executing MLX OP {op}", nameof(TransposeTensorOp));
                
                if (!GetTensor(transpose.ReplyTo, transpose.T, out var transposeT))
                {
                    break;
                }

                var transposeResult = transposeT.Transpose();
                transpose.ReplyTo.SetTensorResponse(transposeResult);
                break;

            case TransposeAxesTensorOp transposeAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(TransposeAxesTensorOp));
                
                if (!GetTensor(transposeAxes.ReplyTo, transposeAxes.T, out var transposeAxesT))
                {
                    break;
                }

                var transposeAxesResult = transposeAxesT.Transpose(transposeAxes.Axes);
                transposeAxes.ReplyTo.SetTensorResponse(transposeAxesResult);
                break;

            case SwapAxesTensorOp swapAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(SwapAxesTensorOp));
                
                if (!GetTensor(swapAxes.ReplyTo, swapAxes.T, out var swapAxesT))
                {
                    break;
                }

                var swapAxesResult = swapAxesT.SwapAxes(swapAxes.A, swapAxes.B);
                swapAxes.ReplyTo.SetTensorResponse(swapAxesResult);
                break;

            case MoveAxisTensorOp moveAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(MoveAxisTensorOp));
                
                if (!GetTensor(moveAxis.ReplyTo, moveAxis.T, out var moveAxisT))
                {
                    break;
                }

                var moveAxisResult = moveAxisT.MoveAxis(moveAxis.Src, moveAxis.Dest);
                moveAxis.ReplyTo.SetTensorResponse(moveAxisResult);
                break;

            case DiagTensorOp diag:
                logger.LogTrace("Executing MLX OP {op}", nameof(DiagTensorOp));
                
                if (!GetTensor(diag.ReplyTo, diag.T, out var diagT))
                {
                    break;
                }

                var diagResult = diagT.Diag(diag.Diagonal);
                diag.ReplyTo.SetTensorResponse(diagResult);
                break;

            case ReshapeTensorOp reshape:
                logger.LogTrace("Executing MLX OP {op}", nameof(ReshapeTensorOp));
                
                if (!GetTensor(reshape.ReplyTo, reshape.T, out var reshapeT))
                {
                    break;
                }

                var reshapeResult = reshapeT.Reshape(reshape.Shape);
                reshape.ReplyTo.SetTensorResponse(reshapeResult);
                break;

            case FlattenTensorOp flatten:
                logger.LogTrace("Executing MLX OP {op}", nameof(FlattenTensorOp));
                
                if (!GetTensor(flatten.ReplyTo, flatten.T, out var flattenT))
                {
                    break;
                }

                var flattenResult = flattenT.Flatten(flatten.StartAxis, flatten.EndAxis);
                flatten.ReplyTo.SetTensorResponse(flattenResult);
                break;

            case ExpandDimsTensorOp expandDims:
                logger.LogTrace("Executing MLX OP {op}", nameof(ExpandDimsTensorOp));
                
                if (!GetTensor(expandDims.ReplyTo, expandDims.T, out var expandDimsT))
                {
                    break;
                }

                var expandDimsResult = expandDimsT.ExpandDims(expandDims.Axis);
                expandDims.ReplyTo.SetTensorResponse(expandDimsResult);
                break;

            case ExpandDimsAxesTensorOp expandDimsAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(ExpandDimsAxesTensorOp));
                
                if (!GetTensor(expandDimsAxes.ReplyTo, expandDimsAxes.T, out var expandDimsAxesT))
                {
                    break;
                }

                var expandDimsAxesResult = expandDimsAxesT.ExpandDims(expandDimsAxes.Axes);
                expandDimsAxes.ReplyTo.SetTensorResponse(expandDimsAxesResult);
                break;

            case BroadcastToTensorOp broadcastTo:
                logger.LogTrace("Executing MLX OP {op}", nameof(BroadcastToTensorOp));
                
                if (!GetTensor(broadcastTo.ReplyTo, broadcastTo.T, out var broadcastToT))
                {
                    break;
                }

                var broadcastToResult = broadcastToT.BroadcastTo(broadcastTo.Shape);
                broadcastTo.ReplyTo.SetTensorResponse(broadcastToResult);
                break;

            case SliceTensorOp slice:
                logger.LogTrace("Executing MLX OP {op}", nameof(SliceTensorOp));
                
                if (!GetTensor(slice.ReplyTo, slice.T, out var sliceT))
                {
                    break;
                }

                var sliceResult = sliceT.Slice(slice.Start, slice.Stop, slice.Strides);
                slice.ReplyTo.SetTensorResponse(sliceResult);
                break;

            case DynamicSliceTensorOp dynamicSlice:
                logger.LogTrace("Executing MLX OP {op}", nameof(DynamicSliceTensorOp));
                
                if (!GetTensor(dynamicSlice.ReplyTo, dynamicSlice.T, out var dsT))
                {
                    break;
                }
                if (!GetTensor(dynamicSlice.ReplyTo, dynamicSlice.Start, out var dsS))
                {
                    break;
                }
                
                var dynamicSliceResult = dsT.DynamicSlice(dsS, dynamicSlice.Axes, dynamicSlice.SliceSices);
                dynamicSlice.ReplyTo.SetTensorResponse(dynamicSliceResult);
                break;

            case SliceUpdateTensorOp sliceUpdate:
                logger.LogTrace("Executing MLX OP {op}", nameof(SliceUpdateTensorOp));
                
                if (!GetTensor(sliceUpdate.ReplyTo, sliceUpdate.T, out var suT))
                {
                    break;
                }
                if (!GetTensor(sliceUpdate.ReplyTo, sliceUpdate.Update, out var suU))
                {
                    break;
                }
                
                var sliceUpdateResult = suT.SliceUpdate(suU, sliceUpdate.Start, sliceUpdate.Stop, sliceUpdate.Strides);
                sliceUpdate.ReplyTo.SetTensorResponse(sliceUpdateResult);
                break;

            case TakeTensorOp take:
                logger.LogTrace("Executing MLX OP {op}", nameof(TakeTensorOp));
                
                if (!GetTensor(take.ReplyTo, take.T, out var takeT))
                {
                    break;
                }
                if (!GetTensor(take.ReplyTo, take.Indices, out var takeIndices))
                {
                    break;
                }
                
                var takeResult = takeT.Take(takeIndices);
                take.ReplyTo.SetTensorResponse(takeResult);
                break;

            case TakeAxisTensorOp takeAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(TakeAxisTensorOp));
                
                if (!GetTensor(takeAxis.ReplyTo, takeAxis.T, out var takeAxisT))
                {
                    break;
                }
                if (!GetTensor(takeAxis.ReplyTo, takeAxis.Indices, out var takeAxisIndices))
                {
                    break;
                }
                
                var takeAxisResult = takeAxisT.Take(takeAxisIndices, takeAxis.Axis);
                takeAxis.ReplyTo.SetTensorResponse(takeAxisResult);
                break;

            case TakeAlongAxisTensorOp takeAlongAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(TakeAlongAxisTensorOp));
                
                if (!GetTensor(takeAlongAxis.ReplyTo, takeAlongAxis.T, out var takeAlongAxisT))
                {
                    break;
                }
                if (!GetTensor(takeAlongAxis.ReplyTo, takeAlongAxis.Indices, out var takeAlongAxisIndices))
                {
                    break;
                }
                
                var takeAlongAxisResult = takeAlongAxisT.TakeAlongAxis(takeAlongAxisIndices, takeAlongAxis.Axis);
                takeAlongAxis.ReplyTo.SetTensorResponse(takeAlongAxisResult);
                break;

            case GatherTensorOp gather:
                logger.LogTrace("Executing MLX OP {op}", nameof(GatherTensorOp));
                
                if (!GetTensor(gather.ReplyTo, gather.T, out var gatherT))
                {
                    break;
                }
                var gatherIndices = GetTensors(gather.ReplyTo, gather.Indices);
                if (gatherIndices is null)
                {
                    break;
                }
                
                var gatherResult = gatherT.Gather(gatherIndices, gather.Axes, gather.SliceSices);
                gather.ReplyTo.SetTensorResponse(gatherResult);
                break;

            case SumTensorOp sum:
                logger.LogTrace("Executing MLX OP {op}", nameof(SumTensorOp));
                
                if (!GetTensor(sum.ReplyTo, sum.T, out var sumT))
                {
                    break;
                }

                var sumResult = sumT.Sum(sum.KeepDims);
                sum.ReplyTo.SetTensorResponse(sumResult);
                break;

            case SumAxisTensorOp sumAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(SumAxisTensorOp));
                
                if (!GetTensor(sumAxis.ReplyTo, sumAxis.T, out var sumAxisT))
                {
                    break;
                }

                var sumAxisResult = sumAxisT.Sum(sumAxis.Axis, sumAxis.KeepDims);
                sumAxis.ReplyTo.SetTensorResponse(sumAxisResult);
                break;

            case SumAxesTensorOp sumAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(SumAxesTensorOp));
                
                if (!GetTensor(sumAxes.ReplyTo, sumAxes.T, out var sumAxesT))
                {
                    break;
                }

                var sumAxesResult = sumAxesT.Sum(sumAxes.Axes, sumAxes.KeepDims);
                sumAxes.ReplyTo.SetTensorResponse(sumAxesResult);
                break;

            case MinTensorOp min:
                logger.LogTrace("Executing MLX OP {op}", nameof(MinTensorOp));
                
                if (!GetTensor(min.ReplyTo, min.T, out var minT))
                {
                    break;
                }

                var minResult = minT.Min(min.KeepDims);
                min.ReplyTo.SetTensorResponse(minResult);
                break;

            case MinAxisTensorOp minAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(MinAxisTensorOp));
                
                if (!GetTensor(minAxis.ReplyTo, minAxis.T, out var minAxisT))
                {
                    break;
                }

                var minAxisResult = minAxisT.Min(minAxis.Axis, minAxis.KeepDims);
                minAxis.ReplyTo.SetTensorResponse(minAxisResult);
                break;

            case MinAxesTensorOp minAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(MinAxesTensorOp));
                
                if (!GetTensor(minAxes.ReplyTo, minAxes.T, out var minAxesT))
                {
                    break;
                }

                var minAxesResult = minAxesT.Min(minAxes.Axes, minAxes.KeepDims);
                minAxes.ReplyTo.SetTensorResponse(minAxesResult);
                break;

            case MaxTensorOp max:
                logger.LogTrace("Executing MLX OP {op}", nameof(MaxTensorOp));
                
                if (!GetTensor(max.ReplyTo, max.T, out var maxT))
                {
                    break;
                }

                var maxResult = maxT.Max(max.KeepDims);
                max.ReplyTo.SetTensorResponse(maxResult);
                break;

            case MaxAxisTensorOp maxAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(MaxAxisTensorOp));
                
                if (!GetTensor(maxAxis.ReplyTo, maxAxis.T, out var maxAxisT))
                {
                    break;
                }

                var maxAxisResult = maxAxisT.Max(maxAxis.Axis, maxAxis.KeepDims);
                maxAxis.ReplyTo.SetTensorResponse(maxAxisResult);
                break;

            case MaxAxesTensorOp maxAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(MaxAxesTensorOp));
                
                if (!GetTensor(maxAxes.ReplyTo, maxAxes.T, out var maxAxesT))
                {
                    break;
                }

                var maxAxesResult = maxAxesT.Max(maxAxes.Axes, maxAxes.KeepDims);
                maxAxes.ReplyTo.SetTensorResponse(maxAxesResult);
                break;

            case MeanTensorOp mean:
                logger.LogTrace("Executing MLX OP {op}", nameof(MeanTensorOp));
                
                if (!GetTensor(mean.ReplyTo, mean.T, out var meanT))
                {
                    break;
                }

                var meanResult = meanT.Mean(mean.KeepDims);
                mean.ReplyTo.SetTensorResponse(meanResult);
                break;

            case MeanAxisTensorOp meanAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(MeanAxisTensorOp));
                
                if (!GetTensor(meanAxis.ReplyTo, meanAxis.T, out var meanAxisT))
                {
                    break;
                }

                var meanAxisResult = meanAxisT.Mean(meanAxis.Axis, meanAxis.KeepDims);
                meanAxis.ReplyTo.SetTensorResponse(meanAxisResult);
                break;

            case MeanAxesTensorOp meanAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(MeanAxesTensorOp));
                
                if (!GetTensor(meanAxes.ReplyTo, meanAxes.T, out var meanAxesT))
                {
                    break;
                }

                var meanAxesResult = meanAxesT.Mean(meanAxes.Axes, meanAxes.KeepDims);
                meanAxes.ReplyTo.SetTensorResponse(meanAxesResult);
                break;

            case StdTensorOp std:
                logger.LogTrace("Executing MLX OP {op}", nameof(StdTensorOp));
                
                if (!GetTensor(std.ReplyTo, std.T, out var stdT))
                {
                    break;
                }

                var stdResult = stdT.Std(std.Ddof, std.KeepDims);
                std.ReplyTo.SetTensorResponse(stdResult);
                break;

            case StdAxisTensorOp stdAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(StdAxisTensorOp));
                
                if (!GetTensor(stdAxis.ReplyTo, stdAxis.T, out var stdAxisT))
                {
                    break;
                }

                var stdAxisResult = stdAxisT.Std(stdAxis.Axis, stdAxis.Ddof, stdAxis.KeepDims);
                stdAxis.ReplyTo.SetTensorResponse(stdAxisResult);
                break;

            case StdAxesTensorOp stdAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(StdAxesTensorOp));
                
                if (!GetTensor(stdAxes.ReplyTo, stdAxes.T, out var stdAxesT))
                {
                    break;
                }

                var stdAxesResult = stdAxesT.Std(stdAxes.Axes, stdAxes.Ddof, stdAxes.KeepDims);
                stdAxes.ReplyTo.SetTensorResponse(stdAxesResult);
                break;

            case ArgMinTensorOp argMin:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArgMinTensorOp));
                
                if (!GetTensor(argMin.ReplyTo, argMin.T, out var argMinT))
                {
                    break;
                }

                var argMinResult = argMinT.ArgMin(argMin.KeepDims);
                argMin.ReplyTo.SetTensorResponse(argMinResult);
                break;

            case ArgMinAxisTensorOp argMinAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArgMinAxisTensorOp));
                
                if (!GetTensor(argMinAxis.ReplyTo, argMinAxis.T, out var argMinAxisT))
                {
                    break;
                }

                var argMinAxisResult = argMinAxisT.ArgMin(argMinAxis.Axis, argMinAxis.KeepDims);
                argMinAxis.ReplyTo.SetTensorResponse(argMinAxisResult);
                break;

            case ArgMaxTensorOp argMax:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArgMaxTensorOp));
                
                if (!GetTensor(argMax.ReplyTo, argMax.T, out var argMaxT))
                {
                    break;
                }

                var argMaxResult = argMaxT.ArgMax(argMax.KeepDims);
                argMax.ReplyTo.SetTensorResponse(argMaxResult);
                break;

            case ArgMaxAxisTensorOp argMaxAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(ArgMaxAxisTensorOp));
                
                if (!GetTensor(argMaxAxis.ReplyTo, argMaxAxis.T, out var argMaxAxisT))
                {
                    break;
                }

                var argMaxAxisResult = argMaxAxisT.ArgMax(argMaxAxis.Axis, argMaxAxis.KeepDims);
                argMaxAxis.ReplyTo.SetTensorResponse(argMaxAxisResult);
                break;

            case VarianceTensorOp variance:
                logger.LogTrace("Executing MLX OP {op}", nameof(VarianceTensorOp));
                
                if (!GetTensor(variance.ReplyTo, variance.T, out var varianceT))
                {
                    break;
                }

                var varianceResult = varianceT.Variance(variance.KeepDims, variance.Ddof);
                variance.ReplyTo.SetTensorResponse(varianceResult);
                break;

            case AllTensorOp all:
                logger.LogTrace("Executing MLX OP {op}", nameof(AllTensorOp));
                
                if (!GetTensor(all.ReplyTo, all.T, out var allT))
                {
                    break;
                }

                var allResult = allT.All(all.KeepDims);
                all.ReplyTo.SetTensorResponse(allResult);
                break;

            case AllAxisTensorOp allAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(AllAxisTensorOp));
                
                if (!GetTensor(allAxis.ReplyTo, allAxis.T, out var allAxisT))
                {
                    break;
                }

                var allAxisResult = allAxisT.All(allAxis.Axis, allAxis.KeepDims);
                allAxis.ReplyTo.SetTensorResponse(allAxisResult);
                break;

            case AllAxesTensorOp allAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(AllAxesTensorOp));
                
                if (!GetTensor(allAxes.ReplyTo, allAxes.T, out var allAxesT))
                {
                    break;
                }

                var allAxesResult = allAxesT.All(allAxes.Axes, allAxes.KeepDims);
                allAxes.ReplyTo.SetTensorResponse(allAxesResult);
                break;

            case AnyTensorOp any:
                logger.LogTrace("Executing MLX OP {op}", nameof(AnyTensorOp));
                
                if (!GetTensor(any.ReplyTo, any.T, out var anyT))
                {
                    break;
                }

                var anyResult = anyT.Any(any.KeepDims);
                any.ReplyTo.SetTensorResponse(anyResult);
                break;

            case AnyAxisTensorOp anyAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(AnyAxisTensorOp));
                
                if (!GetTensor(anyAxis.ReplyTo, anyAxis.T, out var anyAxisT))
                {
                    break;
                }

                var anyAxisResult = anyAxisT.Any(anyAxis.Axis, anyAxis.KeepDims);
                anyAxis.ReplyTo.SetTensorResponse(anyAxisResult);
                break;

            case AnyAxesTensorOp anyAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(AnyAxesTensorOp));
                
                if (!GetTensor(anyAxes.ReplyTo, anyAxes.T, out var anyAxesT))
                {
                    break;
                }

                var anyAxesResult = anyAxesT.Any(anyAxes.Axes, anyAxes.KeepDims);
                anyAxes.ReplyTo.SetTensorResponse(anyAxesResult);
                break;

            case WhereTensorOp where:
                logger.LogTrace("Executing MLX OP {op}", nameof(WhereTensorOp));
                
                if (!GetTensor(where.ReplyTo, where.Cond, out var whereCond))
                {
                    break;
                }
                if (!GetTensor(where.ReplyTo, where.IfTrue, out var whereIfTrue))
                {
                    break;
                }
                if (!GetTensor(where.ReplyTo, where.IfFalse, out var whereIfFalse))
                {
                    break;
                }
                
                var whereResult = whereCond.Where(whereIfTrue, whereIfFalse);
                where.ReplyTo.SetTensorResponse(whereResult);
                break;

            case MinimumTensorOp minimum:
                logger.LogTrace("Executing MLX OP {op}", nameof(MinimumTensorOp));
                
                if (!GetTensor(minimum.ReplyTo, minimum.A, out var minimumA))
                {
                    break;
                }
                if (!GetTensor(minimum.ReplyTo, minimum.B, out var minimumB))
                {
                    break;
                }
                
                var minimumResult = minimumA.Minimum(minimumB);
                minimum.ReplyTo.SetTensorResponse(minimumResult);
                break;

            case MaximumTensorOp maximum:
                logger.LogTrace("Executing MLX OP {op}", nameof(MaximumTensorOp));
                
                if (!GetTensor(maximum.ReplyTo, maximum.A, out var maximumA))
                {
                    break;
                }
                if (!GetTensor(maximum.ReplyTo, maximum.B, out var maximumB))
                {
                    break;
                }
                
                var maximumResult = maximumA.Maximum(maximumB);
                maximum.ReplyTo.SetTensorResponse(maximumResult);
                break;

            case TopKTensorOp topK:
                logger.LogTrace("Executing MLX OP {op}", nameof(TopKTensorOp));
                
                if (!GetTensor(topK.ReplyTo, topK.T, out var topKT))
                {
                    break;
                }

                var topKResult = topKT.TopK(topK.K);
                topK.ReplyTo.SetTensorResponse(topKResult);
                break;

            case TopKAxisTensorOp topKAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(TopKAxisTensorOp));
                
                if (!GetTensor(topKAxis.ReplyTo, topKAxis.T, out var topKAxisT))
                {
                    break;
                }

                var topKAxisResult = topKAxisT.TopK(topKAxis.K, topKAxis.Axis);
                topKAxis.ReplyTo.SetTensorResponse(topKAxisResult);
                break;

            case ZerosLikeTensorOp zerosLike:
                logger.LogTrace("Executing MLX OP {op}", nameof(ZerosLikeTensorOp));
                
                if (!GetTensor(zerosLike.ReplyTo, zerosLike.T, out var zerosLikeT))
                {
                    break;
                }

                var zerosLikeResult = zerosLikeT.ZerosLike();
                zerosLike.ReplyTo.SetTensorResponse(zerosLikeResult);
                break;

            case OnesLikeTensorOp onesLike:
                logger.LogTrace("Executing MLX OP {op}", nameof(OnesLikeTensorOp));
                
                if (!GetTensor(onesLike.ReplyTo, onesLike.T, out var onesLikeT))
                {
                    break;
                }

                var onesLikeResult = onesLikeT.OnesLike();
                onesLike.ReplyTo.SetTensorResponse(onesLikeResult);
                break;

            case SigmoidTensorOp sigmoid:
                logger.LogTrace("Executing MLX OP {op}", nameof(SigmoidTensorOp));
                
                if (!GetTensor(sigmoid.ReplyTo, sigmoid.T, out var sigmoidT))
                {
                    break;
                }

                var sigmoidResult = sigmoidT.Sigmoid();
                sigmoid.ReplyTo.SetTensorResponse(sigmoidResult);
                break;

            case SoftmaxTensorOp softmax:
                logger.LogTrace("Executing MLX OP {op}", nameof(SoftmaxTensorOp));
                
                if (!GetTensor(softmax.ReplyTo, softmax.T, out var softmaxT))
                {
                    break;
                }

                var softmaxResult = softmaxT.Softmax(softmax.Precise);
                softmax.ReplyTo.SetTensorResponse(softmaxResult);
                break;

            case SoftmaxAxisTensorOp softmaxAxis:
                logger.LogTrace("Executing MLX OP {op}", nameof(SoftmaxAxisTensorOp));
                
                if (!GetTensor(softmaxAxis.ReplyTo, softmaxAxis.T, out var softmaxAxisT))
                {
                    break;
                }

                var softmaxAxisResult = softmaxAxisT.Softmax(softmaxAxis.Axis, softmaxAxis.Precise);
                softmaxAxis.ReplyTo.SetTensorResponse(softmaxAxisResult);
                break;

            case SoftmaxAxesTensorOp softmaxAxes:
                logger.LogTrace("Executing MLX OP {op}", nameof(SoftmaxAxesTensorOp));
                
                if (!GetTensor(softmaxAxes.ReplyTo, softmaxAxes.T, out var softmaxAxesT))
                {
                    break;
                }

                var softmaxAxesResult = softmaxAxesT.Softmax(softmaxAxes.Axes, softmaxAxes.Precise);
                softmaxAxes.ReplyTo.SetTensorResponse(softmaxAxesResult);
                break;

            case ErfTensorOp erf:
                logger.LogTrace("Executing MLX OP {op}", nameof(ErfTensorOp));
                
                if (!GetTensor(erf.ReplyTo, erf.T, out var erfT))
                {
                    break;
                }

                var erfResult = erfT.Erf();
                erf.ReplyTo.SetTensorResponse(erfResult);
                break;

            case ErfInvTensorOp erfInv:
                logger.LogTrace("Executing MLX OP {op}", nameof(ErfInvTensorOp));
                
                if (!GetTensor(erfInv.ReplyTo, erfInv.T, out var erfInvT))
                {
                    break;
                }

                var erfInvResult = erfInvT.ErfInv();
                erfInv.ReplyTo.SetTensorResponse(erfInvResult);
                break;
        }
    }

    private void HandleTensorArrayOpCommand(TensorArrayOpCommand command)
    {
        switch (command)
        {
            case SplitTensorOp split:
                break;

            case SplitIndicesTensorOp splitIndices:
                break;
        }
    }

    private bool GetTensor(ReplyTo replyTo, Guid id, [NotNullWhen(true)] out MlxTensor? tensor)
    {
        tensor = manager.Get(id);
        if (tensor is null)
        {
            replyTo.SetErrorResponse($"Tensor with ID '{id}' not found'.");
            
            return false;
        }

        return true;
    }

    private MlxTensor[]? GetTensors(ReplyTo replyTo, List<Guid> ids)
    {
        var tensors = manager.Get(ids);
        if (tensors.Length != ids.Count)
        {
            replyTo.SetErrorResponse($"Not all Tensors found.");

            return null;
        }
        
        return tensors;
    }
}