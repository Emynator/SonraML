using SonraML.Backend.MLX.Extensions;
using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Backend.MLX.Managed;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.ExecutionManagement;

internal unsafe class MlxTensor : IDisposable
{
    private readonly MlxTensorManager tm;
    private readonly MlxStream stream;
    
    public readonly ManagedMlxArray Array;

    #region Ctors

    public MlxTensor(MlxTensorManager tm, MlxStream stream)
    {
        Array = new ManagedMlxArray();
        this.tm = tm;
        this.stream = stream;
    }

    public MlxTensor(MlxTensorManager tm, MlxStream stream, TensorShape shape)
    {
        Array = new ManagedMlxArray();
        this.tm = tm;
        this.stream = stream;
    }

    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<bool> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<byte> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<ushort> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<uint> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<ulong> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<sbyte> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<short> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<int> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<long> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<Half> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<float> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Memory<double> array, TensorShape shape)
    {
        this.Array = new ManagedMlxArray(array, shape);
        this.tm = tm;
        this.stream = stream;
    }

    public MlxTensor(MlxTensorManager tm, MlxStream stream, bool scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, byte scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, ushort scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, uint scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, ulong scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, sbyte scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, short scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, int scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, long scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, Half scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, float scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }
    
    public MlxTensor(MlxTensorManager tm, MlxStream stream, double scalar)
    {
        Array = new ManagedMlxArray(scalar);
        this.tm = tm;
        this.stream = stream;
    }

    #endregion

    public TensorShape Shape => Array.GetShape();

    public bool IsScalar => Interop.MlxArray.NDim(Array.Array) == 0;
    
    public Guid Id { get; init; } = Guid.NewGuid();

    public DType Type => Array.Type;
    
    #region ObjectMethods

    public void Dispose()
    {
        Array.Dispose();
    }

    public IEnumerator<T> GetEnumerator<T>() where T : struct
    {
        return Array.GetEnumerator<T>();
    }

    public MlxTensor Copy()
    {
        var result = tm.CreateEmpty();
        result.Array.CopyFrom(Array);

        return result;
    }

    public void CopyFrom(MlxTensor other)
    {
        Array.CopyFrom(other.Array);
    }

    public MlxTensor ConvertTo(DType type)
    {
        var result = tm.CreateEmpty();
        MlxOps.AsType(in result.Array.Array, Array.Array, type, stream);

        return result;
    }

    public override string ToString()
    {
        return Array.ToString();
    }

    #endregion

    #region TensorOps

    #region ArithmeticOps

    public void EnsureCompute()
    {
        Array.Eval();
    }

    public MlxTensor Add(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.Add(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    public MlxTensor Sub(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.Subtract(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    public MlxTensor Mul(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.Multiply(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    public MlxTensor Rec()
    {
        var result = tm.CreateEmpty();
        MlxOps.Reciprocal(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Div(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.Divide(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    public MlxTensor Mod(MlxTensor other)
    {
        using var vec = new ManagedMlxVectorArray();
        MlxOps.DivMod(in vec.Vector, Array.Array, other.Array.Array, stream);
        var res = vec.Get(1);
        
        var result = tm.CreateEmpty();
        result.Array.CopyFrom(res);
        res.Dispose();
        
        return result;
    }

    public MlxTensor Rem(MlxTensor other)
    {
        using var vec = new ManagedMlxVectorArray();
        MlxOps.DivMod(in vec.Vector, Array.Array, other.Array.Array, stream);
        var res = vec.Get(0);
        
        var result = tm.CreateEmpty();
        result.Array.CopyFrom(res);
        res.Dispose();
        
        return result;
    }

    public MlxTensor Neg()
    {
        var result = tm.CreateEmpty();
        MlxOps.Negative(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Abs()
    {
        var result = tm.CreateEmpty();
        MlxOps.Abs(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Sign()
    {
        var result = tm.CreateEmpty();
        MlxOps.Sign(in result.Array.Array, Array.Array, stream);
        
        return result;
    }
    
    #endregion

    #region LogicalOps

    public MlxTensor Equal(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.Equal(in result.Array.Array, Array.Array, other.Array.Array, stream);

        return result;
    }

    public MlxTensor NotEqual(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.NotEqual(in result.Array.Array, Array.Array, other.Array.Array, stream);

        return result;
    }

    public MlxTensor Less(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.Less(in result.Array.Array, Array.Array, other.Array.Array, stream);

        return result;
    }

    public MlxTensor LessEqual(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.LessEqual(in result.Array.Array, Array.Array, other.Array.Array, stream);

        return result;
    }

    public MlxTensor Greater(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.Greater(in result.Array.Array, Array.Array, other.Array.Array, stream);

        return result;
    }

    public MlxTensor GreaterEqual(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.GreaterEqual(in result.Array.Array, Array.Array, other.Array.Array, stream);

        return result;
    }

    public MlxTensor And(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.LogicalAnd(in result.Array.Array, Array.Array, other.Array.Array, stream);

        return result;
    }

    public MlxTensor Or(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.LogicalOr(in result.Array.Array, Array.Array, other.Array.Array, stream);

        return result;
    }

    public MlxTensor Not()
    {
        var result = tm.CreateEmpty();
        MlxOps.LogicalNot(in result.Array.Array, Array.Array, stream);

        return result;
    }

    public MlxTensor IsNAN()
    {
        var result = tm.CreateEmpty();
        MlxOps.IsNAN(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor IsInfinity()
    {
        var result = tm.CreateEmpty();
        MlxOps.IsInf(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor IsFinite()
    {
        var result = tm.CreateEmpty();
        MlxOps.IsFinite(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor IsClose(MlxTensor other, double rTol, double aTol, bool equalNAN)
    {
        var result = tm.CreateEmpty();
        MlxOps.IsClose(in result.Array.Array, Array.Array, other.Array.Array, rTol, aTol, equalNAN, stream);
        
        return result;
    }

    public MlxTensor AllClose(MlxTensor other, double rTol, double aTol, bool equalNAN)
    {
        var result = tm.CreateEmpty();
        MlxOps.AllClose(in result.Array.Array, Array.Array, other.Array.Array, rTol, aTol, equalNAN, stream);
        
        return result;
    }

    #endregion
    
    #region BitwiseOps

    public MlxTensor BitwiseAnd(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.BitwiseAnd(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    public MlxTensor BitwiseOr(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.BitwiseOr(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    public MlxTensor BitwiseXor(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.BitwiseXor(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    public MlxTensor BitwiseNot()
    {
        var result = tm.CreateEmpty();
        MlxOps.BitwiseInvert(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    #endregion
    
    #region ExponentialOps

    public MlxTensor Exp()
    {
        var result = tm.CreateEmpty();
        MlxOps.Exp(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor ExpM1()
    {
        var result = tm.CreateEmpty();
        MlxOps.ExpM1(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Log()
    {
        var result = tm.CreateEmpty();
        MlxOps.Log(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Log10()
    {
        var result = tm.CreateEmpty();
        MlxOps.Log10(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Log2()
    {
        var result = tm.CreateEmpty();
        MlxOps.Log2(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Log1P()
    {
        var result = tm.CreateEmpty();
        MlxOps.Log1P(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Square()
    {
        var result = tm.CreateEmpty();
        MlxOps.Square(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Sqrt()
    {
        var result = tm.CreateEmpty();
        MlxOps.Sqrt(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor RSqrt()
    {
        var result = tm.CreateEmpty();
        MlxOps.RSqrt(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Pow(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.Power(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    public MlxTensor LogSumExp(bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.LogSumExp(in result.Array.Array, Array.Array, keepDims, stream);
        
        return result;
    }

    public MlxTensor LogSumExp(int axis, bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.LogSumExpAxis(in result.Array.Array, Array.Array, axis, keepDims, stream);
        
        return result;
    }

    public MlxTensor LogSumExp(int[] axes, bool keepDims)
    {
        var result = tm.CreateEmpty();
        using var handle = axes.AsMemory().Pin();
        MlxOps.LogSumExpAxes
            (
                in result.Array.Array,
                Array.Array,
                (int*)handle.Pointer,
                (UIntPtr)axes.Length,
                keepDims, stream
                );
        
        return result;
    }

    public MlxTensor LogAddExp(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.LogAddExp(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    #endregion
    
    #region TrigonometricOps

    public MlxTensor Sin()
    {
        var result = tm.CreateEmpty();
        MlxOps.Sin(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor SinH()
    {
        var result = tm.CreateEmpty();
        MlxOps.SinH(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor ArcSin()
    {
        var result = tm.CreateEmpty();
        MlxOps.ArcSin(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor ArcSinH()
    {
        var result = tm.CreateEmpty();
        MlxOps.ArcSinH(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Cos()
    {
        var result = tm.CreateEmpty();
        MlxOps.Cos(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor CosH()
    {
        var result = tm.CreateEmpty();
        MlxOps.CosH(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor ArcCos()
    {
        var result = tm.CreateEmpty();
        MlxOps.ArcCos(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor ArcCosH()
    {
        var result = tm.CreateEmpty();
        MlxOps.ArcCosH(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Tan()
    {
        var result = tm.CreateEmpty();
        MlxOps.Tan(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor TanH()
    {
        var result = tm.CreateEmpty();
        MlxOps.TanH(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor ArcTan()
    {
        var result = tm.CreateEmpty();
        MlxOps.ArcTan(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor ArcTanH()
    {
        var result = tm.CreateEmpty();
        MlxOps.ArcTanH(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor ArcTan2(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.ArcTan2(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    #endregion
    
    #region Rounding

    public MlxTensor Floor()
    {
        var result = tm.CreateEmpty();
        MlxOps.Floor(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Round(int decimals)
    {
        var result = tm.CreateEmpty();
        MlxOps.Round(in result.Array.Array, Array.Array, decimals, stream);
        
        return result;
    }

    public MlxTensor Ceil()
    {
        var result = tm.CreateEmpty();
        MlxOps.Ceil(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Clip(MlxTensor min, MlxTensor max)
    {
        var result = tm.CreateEmpty();
        MlxOps.Clip(in result.Array.Array, Array.Array, min.Array.Array, max.Array.Array, stream);
        
        return result;
    }

    public MlxTensor FloorDiv(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.FloorDivide(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    #endregion

    #region MatrixOps

    public MlxTensor MatMul(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.MatMul(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    public MlxTensor Fma(MlxTensor a, MlxTensor c, float alpha, float beta)
    {
        var result = tm.CreateEmpty();
        MlxOps.AddMM
        (
            in result.Array.Array,
            c.Array.Array,
            Array.Array,
            a.Array.Array,
            alpha,
            beta,
            stream
        );
        
        return result;
    }

    public MlxTensor Transpose()
    {
        var result = tm.CreateEmpty();
        MlxOps.Transpose(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Transpose(int[] axes)
    {
        using var handler = axes.AsMemory().Pin();
        var result = tm.CreateEmpty();
        MlxOps.TransposeAxes(in result.Array.Array, Array.Array, (int*)handler.Pointer, (UIntPtr)axes.Length, stream);
        
        return result;
    }

    public MlxTensor SwapAxes(int a, int b)
    {
        var result = tm.CreateEmpty();
        MlxOps.SwapAxes(in result.Array.Array, Array.Array, a, b, stream);
        
        return result;
    }

    public MlxTensor MoveAxis(int src, int dest)
    {
        var result = tm.CreateEmpty();
        MlxOps.MoveAxis(in result.Array.Array, Array.Array, src, dest, stream);
        
        return result;
    }

    public MlxTensor Diag(int diagonal)
    {
        var result = tm.CreateEmpty();
        MlxOps.Diag(in result.Array.Array, Array.Array, diagonal, stream);
        
        return result;
    }

    #endregion

    #region ShapeOps

    public MlxTensor Reshape(TensorShape shape)
    {
        using var handle = shape.Shape.AsMemory().Pin();
        var result = tm.CreateEmpty();
        MlxOps.Reshape(in result.Array.Array, Array.Array, (int*)handle.Pointer, (UIntPtr)shape.Dimensions, stream);
        
        return result;
    }

    public MlxTensor Flatten(int startAxis, int endAxis)
    {
        var result = tm.CreateEmpty();
        MlxOps.Flatten(in result.Array.Array, Array.Array, startAxis, endAxis, stream);
        
        return result;
    }

    public MlxTensor ExpandDims(int axis)
    {
        var result = tm.CreateEmpty();
        MlxOps.ExpandDims(in result.Array.Array, Array.Array, axis, stream);
        
        return result;
    }

    public MlxTensor ExpandDims(int[] axes)
    {
        using var handle = axes.AsMemory().Pin();
        var result = tm.CreateEmpty();
        MlxOps.ExpandDimsAxes
            (
                in result.Array.Array,
                Array.Array,
                (int*)handle.Pointer,
                (UIntPtr)axes.Length,
                stream
                );
        
        return result;
    }

    public MlxTensor BroadcastTo(TensorShape shape)
    {
        using var handle = shape.Shape.AsMemory().Pin();
        var result = tm.CreateEmpty();
        MlxOps.BroadcastTo(in result.Array.Array, Array.Array, (int*)handle.Pointer, (UIntPtr)shape.Dimensions, stream);
        
        return result;
    }

    #endregion

    #region IndexingOps

    public MlxTensor Slice(int[] start, int[] stop, int[] strides)
    {
        var result = tm.CreateEmpty();
        using var startHandle = start.AsMemory().Pin();
        using var stopHandle = stop.AsMemory().Pin();
        using var stridesHandle = strides.AsMemory().Pin();
        MlxOps.Slice
            (
                in result.Array.Array,
                Array.Array,
                (int*)startHandle.Pointer,
                (UIntPtr)start.Length,
                (int*)stopHandle.Pointer,
                (UIntPtr)stop.Length,
                (int*)stridesHandle.Pointer,
                (UIntPtr)strides.Length,
                stream
                );
        
        return result;
    }

    public MlxTensor DynamicSlice(MlxTensor start, int[] axes, int[] sliceSize)
    {
        var result = tm.CreateEmpty();
        using var axesHandle = axes.AsMemory().Pin();
        using var sliceHandle = sliceSize.AsMemory().Pin();
        MlxOps.SliceDynamic
        (
            in result.Array.Array,
            Array.Array,
            start.Array.Array,
            (int*)axesHandle.Pointer,
            (UIntPtr)axes.Length,
            (int*)sliceHandle.Pointer,
            (UIntPtr)sliceSize.Length,
            stream
        );
        
        return result;
    }

    public MlxTensor SliceUpdate(MlxTensor other, int[] start, int[] stop, int[] strides)
    {
        var result = tm.CreateEmpty();
        using var startHandle = start.AsMemory().Pin();
        using var stopHandle = stop.AsMemory().Pin();
        using var stridesHandle = strides.AsMemory().Pin();
        MlxOps.SliceUpdate
        (
            in result.Array.Array,
            Array.Array,
            other.Array.Array,
            (int*)startHandle.Pointer,
            (UIntPtr)start.Length,
            (int*)stopHandle.Pointer,
            (UIntPtr)stop.Length,
            (int*)stridesHandle.Pointer,
            (UIntPtr)strides.Length,
            stream
        );
        
        return result;
    }

    public MlxTensor Take(MlxTensor indices)
    {
        var result = tm.CreateEmpty();
        MlxOps.Take(in result.Array.Array, Array.Array, indices.Array.Array, stream);
        
        return result;
    }

    public MlxTensor Take(MlxTensor indices, int axis)
    {
        var result = tm.CreateEmpty();
        MlxOps.TakeAxis(in result.Array.Array, Array.Array, indices.Array.Array, axis, stream);
        
        return result;
    }

    public MlxTensor TakeAlongAxis(MlxTensor indices, int axis)
    {
        var result = tm.CreateEmpty();
        MlxOps.TakeAlongAxis(in result.Array.Array, Array.Array, indices.Array.Array, axis, stream);
        
        return result;
    }

    public MlxTensor Gather(MlxTensor[] indices, int[] axes, int[] sliceSices)
    {
        var result = tm.CreateEmpty();
        using var axesHandle = axes.AsMemory().Pin();
        using var sliceHandle = sliceSices.AsMemory().Pin();
        using var vec = new ManagedMlxVectorArray(indices);
        MlxOps.Gather
            (
                in result.Array.Array,
                Array.Array,
                vec.Vector,
                (int*)axesHandle.Pointer,
                (UIntPtr)axes.Length,
                (int*)sliceHandle.Pointer,
                (UIntPtr)sliceSices.Length,
                stream
                );
        
        return result;
    }

    #endregion

    #region SplitOps

    public MlxTensor[] Split(int numSplits, int axis)
    {
        using var vec = new ManagedMlxVectorArray();
        MlxOps.Split(in vec.Vector, this.Array.Array, numSplits, axis, stream);

        var result = new MlxTensor[vec.Size];
        var array = vec.ToArray();
        for (UIntPtr i = 0; i < vec.Size; i++)
        {
            var t = tm.CreateEmpty();
            t.Array.CopyFrom(array[i]);
            result[i] = t;
        }
        
        return result;
    }

    public MlxTensor[] Split(int[] indices, int axis)
    {
        using var vec = new ManagedMlxVectorArray();
        using var indicesHandle = indices.AsMemory().Pin();
        MlxOps.SplitSections
            (
                in vec.Vector,
                this.Array.Array,
                (int*)indicesHandle.Pointer,
                (UIntPtr)indices.Length,
                axis,
                stream
                );

        var result = new MlxTensor[vec.Size];
        var array = vec.ToArray();
        for (UIntPtr i = 0; i < vec.Size; i++)
        {
            var t = tm.CreateEmpty();
            t.Array.CopyFrom(array[i]);
            result[i] = t;
        }
        
        return result;
    }

    #endregion

    #region PredicateOps

    public MlxTensor Sum(bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.Sum(in result.Array.Array, Array.Array, keepDims, stream);
        
        return result;
    }

    public MlxTensor Sum(int axis, bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.SumAxis(in result.Array.Array, Array.Array, axis, keepDims, stream);
        
        return result;
    }

    public MlxTensor Sum(int[] axes, bool keepDims)
    {
        var result = tm.CreateEmpty();
        using var handle = axes.AsMemory().Pin();
        MlxOps.SumAxes
            (
                in result.Array.Array,
                Array.Array,
                (int*)handle.Pointer,
                (UIntPtr)axes.Length,
                keepDims,
                stream
                );
        
        return result;
    }

    public MlxTensor Min(bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.Min(in result.Array.Array, Array.Array, keepDims, stream);
        
        return result;
    }

    public MlxTensor Min(int axis, bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.MinAxis(in result.Array.Array, Array.Array, axis, keepDims, stream);
        
        return result;
    }

    public MlxTensor Min(int[] axes, bool keepDims)
    {
        var result = tm.CreateEmpty();
        using var handle = axes.AsMemory().Pin();
        MlxOps.MinAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            stream
            );
        
        return result;
    }

    public MlxTensor Max(bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.Max(in result.Array.Array, Array.Array, keepDims, stream);
        
        return result;
    }

    public MlxTensor Max(int axis, bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.MaxAxis(in result.Array.Array, Array.Array, axis, keepDims, stream);
        
        return result;
    }

    public MlxTensor Max(int[] axes, bool keepDims)
    {
        var result = tm.CreateEmpty();
        using var handle = axes.AsMemory().Pin();
        MlxOps.MaxAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            stream
        );
        
        return result;
    }

    public MlxTensor Mean(bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.Mean(in result.Array.Array, Array.Array, keepDims, stream);
        
        return result;
    }

    public MlxTensor Mean(int axis, bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.MeanAxis(in result.Array.Array, Array.Array, axis, keepDims, stream);
        
        return result;
    }

    public MlxTensor Mean(int[] axes, bool keepDims)
    {
        var result = tm.CreateEmpty();
        using var handle = axes.AsMemory().Pin();
        MlxOps.MeanAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            stream
        );
        
        return result;
    }

    public MlxTensor Std(int ddof, bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.Std(in result.Array.Array, Array.Array, keepDims, ddof, stream);
        
        return result;
    }

    public MlxTensor Std(int axis, int ddof, bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.StdAxis(in result.Array.Array, Array.Array, axis, keepDims, ddof, stream);
        
        return result;
    }

    public MlxTensor Std(int[] axes, int ddof, bool keepDims)
    {
        var result = tm.CreateEmpty();
        using var handle = axes.AsMemory().Pin();
        MlxOps.StdAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            ddof,
            stream
        );
        
        return result;
    }

    public MlxTensor ArgMin(bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.ArgMin(in result.Array.Array, Array.Array, keepDims, stream);
        
        return result;
    }

    public MlxTensor ArgMin(int axis, bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.ArgMinAxis(in result.Array.Array, Array.Array, axis, keepDims, stream);
        
        return result;
    }

    public MlxTensor ArgMax(bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.ArgMax(in result.Array.Array, Array.Array, keepDims, stream);
        
        return result;
    }

    public MlxTensor ArgMax(int axis, bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.ArgMaxAxis(in result.Array.Array, Array.Array, axis, keepDims, stream);
        
        return result;
    }

    public MlxTensor Variance(bool keepDims, int ddof)
    {
        var result = tm.CreateEmpty();
        MlxOps.Var(in result.Array.Array, Array.Array, keepDims, ddof, stream);
        
        return result;
    }

    #endregion

    #region SelectionOps

    public MlxTensor All(bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.All(in result.Array.Array, Array.Array, keepDims, stream);
        
        return result;
    }

    public MlxTensor All(int axis, bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.AllAxis(in result.Array.Array, Array.Array, axis, keepDims, stream);
        
        return result;
    }

    public MlxTensor All(int[] axes, bool keepDims)
    {
        var result = tm.CreateEmpty();
        using var handle = axes.AsMemory().Pin();
        MlxOps.AllAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            stream
        );
        
        return result;
    }

    public MlxTensor Any(bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.Any(in result.Array.Array, Array.Array, keepDims, stream);
        
        return result;
    }

    public MlxTensor Any(int axis, bool keepDims)
    {
        var result = tm.CreateEmpty();
        MlxOps.AnyAxis(in result.Array.Array, Array.Array, axis, keepDims, stream);
        
        return result;
    }

    public MlxTensor Any(int[] axes, bool keepDims)
    {
        var result = tm.CreateEmpty();
        using var handle = axes.AsMemory().Pin();
        MlxOps.AnyAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            stream
        );
        
        return result;
    }

    public MlxTensor Where(MlxTensor ifTrue, MlxTensor ifFalse)
    {
        var result = tm.CreateEmpty();
        MlxOps.Where(in result.Array.Array, Array.Array, ifTrue.Array.Array, ifFalse.Array.Array, stream);
        
        return result;
    }

    public MlxTensor Minimum(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.Minimum(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    public MlxTensor Maximum(MlxTensor other)
    {
        var result = tm.CreateEmpty();
        MlxOps.Maximum(in result.Array.Array, Array.Array, other.Array.Array, stream);
        
        return result;
    }

    public MlxTensor TopK(int k)
    {
        var result = tm.CreateEmpty();
        MlxOps.TopK(in result.Array.Array, Array.Array, k, stream);
        
        return result;
    }

    public MlxTensor TopK(int k, int axis)
    {
        var result = tm.CreateEmpty();
        MlxOps.TopKAxis(in result.Array.Array, Array.Array, k, axis, stream);
        
        return result;
    }

    #endregion

    #region LikeOps

    public MlxTensor ZerosLike()
    {
        var result = tm.CreateEmpty();
        MlxOps.ZerosLike(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor OnesLike()
    {
        var result = tm.CreateEmpty();
        MlxOps.OnesLike(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    #endregion

    #region NeuralOps

    public MlxTensor Sigmoid()
    {
        var result = tm.CreateEmpty();
        MlxOps.Sigmoid(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor Softmax(bool precise)
    {
        var result = tm.CreateEmpty();
        MlxOps.Softmax(in result.Array.Array, Array.Array, precise, stream);
        
        return result;
    }

    public MlxTensor Softmax(int axis, bool precise)
    {
        var result = tm.CreateEmpty();
        MlxOps.SoftmaxAxis(in result.Array.Array, Array.Array, axis, precise, stream);
        
        return result;
    }

    public MlxTensor Softmax(int[] axes, bool precise)
    {
        var result = tm.CreateEmpty();
        using var handle = axes.AsMemory().Pin();
        MlxOps.SoftmaxAxes
            (
                in result.Array.Array,
                Array.Array,
                (int*)handle.Pointer,
                (UIntPtr)axes.Length,
                precise,
                stream
                );
        
        return result;
    }

    public MlxTensor Erf()
    {
        var result = tm.CreateEmpty();
        MlxOps.Erf(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    public MlxTensor ErfInv()
    {
        var result = tm.CreateEmpty();
        MlxOps.ErfInv(in result.Array.Array, Array.Array, stream);
        
        return result;
    }

    #endregion

    #endregion
}