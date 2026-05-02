using SonraML.Backend.MLX.Extensions;
using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Backend.MLX.Managed;
using SonraML.Core.Exceptions;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.Implementations;

internal unsafe class MlxTensor<T> : Tensor<T> where T : struct
{
    private readonly MlxTensorFactory tf;
    
    public readonly ManagedMlxArray<T> Array;

    #region Ctors

    public MlxTensor(MlxTensorFactory tf, string? name = null)
    {
        Array = new ManagedMlxArray<T>();
        this.tf = tf;
        Name = name ?? "";
        Type = typeof(T);
    }

    public MlxTensor(MlxTensorFactory tf, bool isZero, string? name = null)
    {
        Array = new ManagedMlxArray<T>(isZero);
        this.tf = tf;
        Name = name ?? "";
        Type = typeof(T);
    }

    public MlxTensor(MlxTensorFactory tf, TensorShape shape, string? name = null)
    {
        Array = new ManagedMlxArray<T>();
        this.tf = tf;
        this.shape = shape;
        Name = name ?? "";
        Type = typeof(T);
    }

    public MlxTensor(MlxTensorFactory tf, Memory<T> array, TensorShape shape, string? name = null)
    {
        this.Array = new ManagedMlxArray<T>(array, shape);
        this.tf = tf;
        Name = name ?? "";
        Type = typeof(T);
    }

    public MlxTensor(MlxTensorFactory tf, T scalar, string? name = null)
    {
        Array = new ManagedMlxArray<T>(scalar);
        this.tf = tf;
        Name = name ?? "";
        Type = typeof(T);
    }

    #endregion

    public override TensorShape Shape => Array.GetShape();

    public override bool IsScalar => MlxArray.NDim(Array.Array) == 0;
    
    private MlxStream Stream => tf.Stream.Stream;

    #region ObjectMethods

    public override void Release()
    {
        if (!isReleased)
        {
            isReleased = true;
            Array.Dispose();
        }
    }

    public override bool Equals(Tensor<T>? other)
    {
        if (other is null)
        {
            return false;
        }

        if (other is not MlxTensor<T> o)
        {
            return false;
        }

        if (ReferenceEquals(this, other))
        {
            return true;
        }

        if (IsScalar != other.IsScalar)
        {
            return false;
        }

        if (!Shape.Equals(other.Shape))
        {
            return false;
        }

        using var result = new ManagedMlxArray<bool>();
        MlxOps.ArrayEqual(in result.Array, Array.Array, o.Array.Array, false, Stream);

        return result.GetScalar();
    }

    public override IEnumerator<T> GetEnumerator()
    {
        return Array.GetEnumerator();
    }

    public override object Clone()
    {
        var result = tf.CreateEmpty<T>();
        result.Array.CopyFrom(Array);

        return result;
    }

    public override void CopyFrom(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }
        
        Array.CopyFrom(o.Array);
    }

    public override Tensor<TTarget> ConvertTo<TTarget>()
    {
        var dtype = MlxDType.GetDType<TTarget>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(TTarget));
        }

        var result = tf.CreateEmpty<TTarget>();
        MlxOps.AsType(in result.Array.Array, Array.Array, dtype.Value, Stream);

        return result;
    }

    public override string ToString()
    {
        return Array.ToString();
    }

    internal void SetZero()
    {
        using var shapeHandle = shape?.GetHandle() ?? throw new InvalidOperationException();
        MlxOps.Zeros
        (
            in Array.Array,
            (int*)shapeHandle.Pointer,
            (UIntPtr)shape.Dimensions,
            MlxDType.GetDTypeValid<T>(),
            Stream
        );
    }

    internal void SetOne()
    {
        using var shapeHandle = shape?.GetHandle() ?? throw new InvalidOperationException();
        MlxOps.Ones
        (
            in Array.Array,
            (int*)shapeHandle.Pointer,
            (UIntPtr)shape.Dimensions,
            MlxDType.GetDTypeValid<T>(),
            Stream
        );
    }

    #endregion

    #region TensorOps

    #region ArithmeticOps

    public override void EnsureCompute()
    {
        Array.Eval();
    }

    public override Tensor<T> Add(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Add(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Sub(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Subtract(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Mul(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Multiply(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Rec()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Reciprocal(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Div(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Divide(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Mod(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        using var vec = new ManagedMlxVectorArray<T>();
        MlxOps.DivMod(in vec.Vector, Array.Array, o.Array.Array, Stream);
        var res = vec.Get(1);
        
        var result = tf.CreateEmpty<T>();
        result.Array.CopyFrom(res);
        res.Dispose();
        
        return result;
    }

    public override Tensor<T> Rem(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }
        
        using var vec = new ManagedMlxVectorArray<T>();
        MlxOps.DivMod(in vec.Vector, Array.Array, o.Array.Array, Stream);
        var res = vec.Get(0);
        
        var result = tf.CreateEmpty<T>();
        result.Array.CopyFrom(res);
        res.Dispose();
        
        return result;
    }

    public override Tensor<T> Neg()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Negative(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Abs()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Abs(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Sign()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Sign(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }
    
    #endregion

    #region LogicalOps

    public override Tensor<bool> Equal(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.Equal(in result.Array.Array, Array.Array, o.Array.Array, Stream);

        return result;
    }

    public override Tensor<bool> NotEqual(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.NotEqual(in result.Array.Array, Array.Array, o.Array.Array, Stream);

        return result;
    }

    public override Tensor<bool> Less(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.Less(in result.Array.Array, Array.Array, o.Array.Array, Stream);

        return result;
    }

    public override Tensor<bool> LessEqual(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.LessEqual(in result.Array.Array, Array.Array, o.Array.Array, Stream);

        return result;
    }

    public override Tensor<bool> Greater(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.Greater(in result.Array.Array, Array.Array, o.Array.Array, Stream);

        return result;
    }

    public override Tensor<bool> GreaterEqual(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.GreaterEqual(in result.Array.Array, Array.Array, o.Array.Array, Stream);

        return result;
    }

    public override Tensor<bool> And(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.LogicalAnd(in result.Array.Array, Array.Array, o.Array.Array, Stream);

        return result;
    }

    public override Tensor<bool> Or(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.LogicalOr(in result.Array.Array, Array.Array, o.Array.Array, Stream);

        return result;
    }

    public override Tensor<bool> Not()
    {
        var result = tf.CreateEmpty<bool>();
        MlxOps.LogicalNot(in result.Array.Array, Array.Array, Stream);

        return result;
    }

    public override Tensor<bool> IsNAN()
    {
        var result = tf.CreateEmpty<bool>();
        MlxOps.IsNAN(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<bool> IsInfinity()
    {
        var result = tf.CreateEmpty<bool>();
        MlxOps.IsInf(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<bool> IsFinite()
    {
        var result = tf.CreateEmpty<bool>();
        MlxOps.IsFinite(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<bool> IsClose(Tensor<T> other, double rTol, double aTol, bool equalNAN = false)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }
        
        var result = tf.CreateEmpty<bool>();
        MlxOps.IsClose(in result.Array.Array, Array.Array, o.Array.Array, rTol, aTol, equalNAN, Stream);
        
        return result;
    }

    public override Tensor<bool> AllClose(Tensor<T> other, double rTol, double aTol, bool equalNAN = false)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }
        
        var result = tf.CreateEmpty<bool>();
        MlxOps.AllClose(in result.Array.Array, Array.Array, o.Array.Array, rTol, aTol, equalNAN, Stream);
        
        return result;
    }

    #endregion
    
    #region BitwiseOps

    public override Tensor<T> BitwiseAnd(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.BitwiseAnd(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> BitwiseOr(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.BitwiseOr(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> BitwiseXor(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.BitwiseXor(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> BitwiseNot()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.BitwiseInvert(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    #endregion
    
    #region ExponentialOps

    public override Tensor<T> Exp()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Exp(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> ExpM1()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ExpM1(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Log()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Log(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Log10()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Log10(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Log2()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Log2(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Log1P()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Log1P(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Square()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Square(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Sqrt()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Sqrt(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> RSqrt()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.RSqrt(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Pow(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Power(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> LogSumExp(bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.LogSumExp(in result.Array.Array, Array.Array, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> LogSumExp(int axis, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.LogSumExpAxis(in result.Array.Array, Array.Array, axis, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> LogSumExp(int[] axes, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        using var handle = axes.AsMemory().Pin();
        MlxOps.LogSumExpAxes
            (
                in result.Array.Array,
                Array.Array,
                (int*)handle.Pointer,
                (UIntPtr)axes.Length,
                keepDims, Stream
                );
        
        return result;
    }

    public override Tensor<T> LogAddExp(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }
        
        var result = tf.CreateEmpty<T>();
        MlxOps.LogAddExp(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    #endregion
    
    #region TrigonometricOps

    public override Tensor<T> Sin()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Sin(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> SinH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.SinH(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> ArcSin()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcSin(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> ArcSinH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcSinH(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Cos()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Cos(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> CosH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.CosH(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> ArcCos()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcCos(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> ArcCosH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcCosH(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Tan()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Tan(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> TanH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.TanH(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> ArcTan()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcTan(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> ArcTanH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcTanH(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> ArcTan2(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.ArcTan2(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    #endregion
    
    #region Rounding

    public override Tensor<T> Floor()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Floor(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Round(int decimals)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Round(in result.Array.Array, Array.Array, decimals, Stream);
        
        return result;
    }

    public override Tensor<T> Ceil()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Ceil(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Clip(T min, T max)
    {
        using var sMin = new ManagedMlxArray<T>(min);
        using var sMax = new ManagedMlxArray<T>(max);

        var result = tf.CreateEmpty<T>();
        MlxOps.Clip(in result.Array.Array, Array.Array, sMin.Array, sMax.Array, Stream);
        
        return result;
    }

    public override Tensor<T> FloorDiv(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }
        
        var result = tf.CreateEmpty<T>();
        MlxOps.FloorDivide(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    #endregion

    #region MatrixOps

    public override Tensor<T> MatMul(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.MatMul(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Fma(Tensor<T> a, Tensor<T> c, float alpha = 1.0f, float beta = 1.0f)
    {
        if (a is not MlxTensor<T> mul)
        {
            throw new TensorCompatibilityException();
        }

        if (c is not MlxTensor<T> add)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.AddMM
        (
            in result.Array.Array,
            add.Array.Array,
            Array.Array,
            mul.Array.Array,
            alpha,
            beta,
            Stream
        );
        
        return result;
    }

    public override Tensor<T> Transpose()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Transpose(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Transpose(int[] axes)
    {
        using var handler = axes.AsMemory().Pin();
        var result = tf.CreateEmpty<T>();
        MlxOps.TransposeAxes(in result.Array.Array, Array.Array, (int*)handler.Pointer, (UIntPtr)axes.Length, Stream);
        
        return result;
    }

    public override Tensor<T> SwapAxes(int a, int b)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.SwapAxes(in result.Array.Array, Array.Array, a, b, Stream);
        
        return result;
    }

    public override Tensor<T> MoveAxis(int src, int dest)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.MoveAxis(in result.Array.Array, Array.Array, src, dest, Stream);
        
        return result;
    }

    public override Tensor<T> Diag(int diagonal)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Diag(in result.Array.Array, Array.Array, diagonal, Stream);
        
        return result;
    }

    #endregion

    #region ShapeOps

    public override Tensor<T> Reshape(TensorShape shape)
    {
        using var handle = shape.Shape.AsMemory().Pin();
        var result = tf.CreateEmpty<T>();
        MlxOps.Reshape(in result.Array.Array, Array.Array, (int*)handle.Pointer, (UIntPtr)shape.Size, Stream);
        
        return result;
    }

    public override Tensor<T> Flatten(int startAxis, int endAxis)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Flatten(in result.Array.Array, Array.Array, startAxis, endAxis, Stream);
        
        return result;
    }

    public override Tensor<T> ExpandDims(int axis)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ExpandDims(in result.Array.Array, Array.Array, axis, Stream);
        
        return result;
    }

    public override Tensor<T> ExpandDims(int[] axes)
    {
        using var handle = axes.AsMemory().Pin();
        var result = tf.CreateEmpty<T>();
        MlxOps.ExpandDimsAxes
            (
                in result.Array.Array,
                Array.Array,
                (int*)handle.Pointer,
                (UIntPtr)axes.Length,
                Stream
                );
        
        return result;
    }

    public override Tensor<T> BroadcastTo(TensorShape shape)
    {
        using var handle = shape.Shape.AsMemory().Pin();
        var result = tf.CreateEmpty<T>();
        MlxOps.BroadcastTo(in result.Array.Array, Array.Array, (int*)handle.Pointer, (UIntPtr)shape.Size, Stream);
        
        return result;
    }

    #endregion

    #region IndexingOps

    public override Tensor<T> Slice(int[] start, int[] stop, int[] strides)
    {
        var result = tf.CreateEmpty<T>();
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
                Stream
                );
        
        return result;
    }

    public override Tensor<T> DynamicSlice(Tensor<T> start, int[] axes, int[] sliceSize)
    {
        if (start is not MlxTensor<T> s)
        {
            throw new TensorCompatibilityException();
        }
        
        var result = tf.CreateEmpty<T>();
        using var axesHandle = axes.AsMemory().Pin();
        using var sliceHandle = sliceSize.AsMemory().Pin();
        MlxOps.SliceDynamic
        (
            in result.Array.Array,
            Array.Array,
            s.Array.Array,
            (int*)axesHandle.Pointer,
            (UIntPtr)axes.Length,
            (int*)sliceHandle.Pointer,
            (UIntPtr)sliceSize.Length,
            Stream
        );
        
        return result;
    }

    public override Tensor<T> SliceUpdate(Tensor<T> other, int[] start, int[] stop, int[] strides)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }
        
        var result = tf.CreateEmpty<T>();
        using var startHandle = start.AsMemory().Pin();
        using var stopHandle = stop.AsMemory().Pin();
        using var stridesHandle = strides.AsMemory().Pin();
        MlxOps.SliceUpdate
        (
            in result.Array.Array,
            Array.Array,
            o.Array.Array,
            (int*)startHandle.Pointer,
            (UIntPtr)start.Length,
            (int*)stopHandle.Pointer,
            (UIntPtr)stop.Length,
            (int*)stridesHandle.Pointer,
            (UIntPtr)strides.Length,
            Stream
        );
        
        return result;
    }

    public override Tensor<T> Take(Tensor<T> indices)
    {
        if (indices is not MlxTensor<T> i)
        {
            throw new TensorCompatibilityException();
        }
        
        var result = tf.CreateEmpty<T>();
        MlxOps.Take(in result.Array.Array, Array.Array, i.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Take(Tensor<T> indices, int axis)
    {
        if (indices is not MlxTensor<T> i)
        {
            throw new TensorCompatibilityException();
        }
        
        var result = tf.CreateEmpty<T>();
        MlxOps.TakeAxis(in result.Array.Array, Array.Array, i.Array.Array, axis, Stream);
        
        return result;
    }

    public override Tensor<T> TakeAlongAxis(Tensor<T> indices, int axis)
    {
        if (indices is not MlxTensor<T> i)
        {
            throw new TensorCompatibilityException();
        }
        
        var result = tf.CreateEmpty<T>();
        MlxOps.TakeAlongAxis(in result.Array.Array, Array.Array, i.Array.Array, axis, Stream);
        
        return result;
    }

    public override Tensor<T> Gather(Tensor<T>[] indices, int[] axes, int[] sliceSices)
    {
        if (indices is not MlxTensor<T>[] i)
        {
            throw new TensorCompatibilityException();
        }
        
        var result = tf.CreateEmpty<T>();
        using var axesHandle = axes.AsMemory().Pin();
        using var sliceHandle = sliceSices.AsMemory().Pin();
        using var vec = new ManagedMlxVectorArray<T>(i);
        MlxOps.Gather
            (
                in result.Array.Array,
                Array.Array,
                vec.Vector,
                (int*)axesHandle.Pointer,
                (UIntPtr)axes.Length,
                (int*)sliceHandle.Pointer,
                (UIntPtr)sliceSices.Length,
                Stream
                );
        
        return result;
    }

    #endregion

    #region SplitOps

    public override Tensor<T>[] Split(int numSplits, int axis)
    {
        using var vec = new ManagedMlxVectorArray<T>();
        MlxOps.Split(in vec.Vector, Array.Array, numSplits, axis, Stream);

        var result = new Tensor<T>[vec.Size];
        var array = vec.ToArray();
        for (UIntPtr i = 0; i < vec.Size; i++)
        {
            var t = tf.CreateEmpty<T>();
            t.Array.CopyFrom(array[i]);
            result[i] = t;
        }
        
        return result;
    }

    public override Tensor<T>[] Split(int[] indices, int axis)
    {
        using var vec = new ManagedMlxVectorArray<T>();
        using var indicesHandle = indices.AsMemory().Pin();
        MlxOps.SplitSections
            (
                in vec.Vector,
                Array.Array,
                (int*)indicesHandle.Pointer,
                (UIntPtr)indices.Length,
                axis,
                Stream
                );

        var result = new Tensor<T>[vec.Size];
        var array = vec.ToArray();
        for (UIntPtr i = 0; i < vec.Size; i++)
        {
            var t = tf.CreateEmpty<T>();
            t.Array.CopyFrom(array[i]);
            result[i] = t;
        }
        
        return result;
    }

    #endregion

    #region PredicateOps

    public override Tensor<T> Sum(bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Sum(in result.Array.Array, Array.Array, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> Sum(int axis, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.SumAxis(in result.Array.Array, Array.Array, axis, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> Sum(int[] axes, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        using var handle = axes.AsMemory().Pin();
        MlxOps.SumAxes
            (
                in result.Array.Array,
                Array.Array,
                (int*)handle.Pointer,
                (UIntPtr)axes.Length,
                keepDims,
                Stream
                );
        
        return result;
    }

    public override Tensor<T> Min(bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Min(in result.Array.Array, Array.Array, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> Min(int axis, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.MinAxis(in result.Array.Array, Array.Array, axis, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> Min(int[] axes, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        using var handle = axes.AsMemory().Pin();
        MlxOps.MinAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            Stream
            );
        
        return result;
    }

    public override Tensor<T> Max(bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Max(in result.Array.Array, Array.Array, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> Max(int axis, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.MaxAxis(in result.Array.Array, Array.Array, axis, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> Max(int[] axes, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        using var handle = axes.AsMemory().Pin();
        MlxOps.MaxAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            Stream
        );
        
        return result;
    }

    public override Tensor<T> Mean(bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Mean(in result.Array.Array, Array.Array, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> Mean(int axis, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.MeanAxis(in result.Array.Array, Array.Array, axis, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> Mean(int[] axes, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        using var handle = axes.AsMemory().Pin();
        MlxOps.MeanAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            Stream
        );
        
        return result;
    }

    public override Tensor<T> Std(int ddof, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Std(in result.Array.Array, Array.Array, keepDims, ddof, Stream);
        
        return result;
    }

    public override Tensor<T> Std(int axis, int ddof, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.StdAxis(in result.Array.Array, Array.Array, axis, keepDims, ddof, Stream);
        
        return result;
    }

    public override Tensor<T> Std(int[] axes, int ddof, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        using var handle = axes.AsMemory().Pin();
        MlxOps.StdAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            ddof,
            Stream
        );
        
        return result;
    }

    public override Tensor<T> ArgMin(bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArgMin(in result.Array.Array, Array.Array, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> ArgMin(int axis, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArgMinAxis(in result.Array.Array, Array.Array, axis, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> ArgMax(bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArgMax(in result.Array.Array, Array.Array, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> ArgMax(int axis, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArgMaxAxis(in result.Array.Array, Array.Array, axis, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> Variance(bool keepDims, int ddof)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Var(in result.Array.Array, Array.Array, keepDims, ddof, Stream);
        
        return result;
    }

    #endregion

    #region SelectionOps

    public override Tensor<T> All(bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.All(in result.Array.Array, Array.Array, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> All(int axis, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.AllAxis(in result.Array.Array, Array.Array, axis, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> All(int[] axes, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        using var handle = axes.AsMemory().Pin();
        MlxOps.AllAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            Stream
        );
        
        return result;
    }

    public override Tensor<T> Any(bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Any(in result.Array.Array, Array.Array, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> Any(int axis, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.AnyAxis(in result.Array.Array, Array.Array, axis, keepDims, Stream);
        
        return result;
    }

    public override Tensor<T> Any(int[] axes, bool keepDims = false)
    {
        var result = tf.CreateEmpty<T>();
        using var handle = axes.AsMemory().Pin();
        MlxOps.AnyAxes
        (
            in result.Array.Array,
            Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)axes.Length,
            keepDims,
            Stream
        );
        
        return result;
    }

    public override Tensor<TResult> Where<TResult>
        (
        Tensor<TResult> ifTrue,
        Tensor<TResult> ifFalse
        ) where TResult : struct
    {
        if (ifTrue is not MlxTensor<TResult> ifT)
        {
            throw new TensorCompatibilityException();
        }
        
        if (ifFalse is not MlxTensor<TResult> ifF)
        {
            throw new TensorCompatibilityException();
        }

        if (typeof(T) != typeof(bool))
        {
            throw new InvalidOperationException("Where requires 'Tensor<bool>'.");
        }
        
        var result = tf.CreateEmpty<TResult>();
        MlxOps.Where(in result.Array.Array, Array.Array, ifT.Array.Array, ifF.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Minimum(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Minimum(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Maximum(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Maximum(in result.Array.Array, Array.Array, o.Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> TopK(int k)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.TopK(in result.Array.Array, Array.Array, k, Stream);
        
        return result;
    }

    public override Tensor<T> TopK(int k, int axis)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.TopKAxis(in result.Array.Array, Array.Array, k, axis, Stream);
        
        return result;
    }

    #endregion

    #region LikeOps

    public override Tensor<T> ZerosLike()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ZerosLike(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> OnesLike()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.OnesLike(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    #endregion

    #region NeuralOps

    public override Tensor<T> Sigmoid()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Sigmoid(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> Softmax(bool precise = true)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Softmax(in result.Array.Array, Array.Array, precise, Stream);
        
        return result;
    }

    public override Tensor<T> Softmax(int axis, bool precise = true)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.SoftmaxAxis(in result.Array.Array, Array.Array, axis, precise, Stream);
        
        return result;
    }

    public override Tensor<T> Softmax(int[] axes, bool precise = true)
    {
        var result = tf.CreateEmpty<T>();
        using var handle = axes.AsMemory().Pin();
        MlxOps.SoftmaxAxes
            (
                in result.Array.Array,
                Array.Array,
                (int*)handle.Pointer,
                (UIntPtr)axes.Length,
                precise,
                Stream
                );
        
        return result;
    }

    public override Tensor<T> Erf()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Erf(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    public override Tensor<T> ErfInv()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ErfInv(in result.Array.Array, Array.Array, Stream);
        
        return result;
    }

    #endregion

    #endregion
}