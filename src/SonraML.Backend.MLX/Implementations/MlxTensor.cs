using SonraML.Backend.MLX.Extensions;
using SonraML.Backend.MLX.Implementations;
using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Backend.MLX.Managed;
using SonraML.Core.Exceptions;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX;

internal unsafe class MlxTensor<T> : Tensor<T> where T : struct
{
    private readonly ManagedMlxArray<T> array;
    private readonly MlxTensorFactory tf;
    private readonly MlxStream stream;

    #region Ctors

    public MlxTensor(MlxTensorFactory tf, MlxStream stream, string? name = null)
    {
        array = new ManagedMlxArray<T>();
        this.tf = tf;
        this.stream = stream;
        Name = name ?? "";
        Type = typeof(T);
    }

    public MlxTensor(MlxTensorFactory tf, MlxStream stream, bool isZero, string? name = null)
    {
        array = new ManagedMlxArray<T>(isZero);
        this.tf = tf;
        this.stream = stream;
        Name = name ?? "";
        Type = typeof(T);
        IsScalar = true;
    }

    public MlxTensor(MlxTensorFactory tf, MlxStream stream, TensorShape shape, string? name = null)
    {
        array = new ManagedMlxArray<T>();
        this.tf = tf;
        this.stream = stream;
        this.shape = shape;
        Name = name ?? "";
        Type = typeof(T);
        IsScalar = shape.Dimensions == 0;
    }

    public MlxTensor(MlxTensorFactory tf, MlxStream stream, Memory<T> array, TensorShape shape, string? name = null)
    {
        this.array = new ManagedMlxArray<T>(array, shape);
        this.tf = tf;
        this.stream = stream;
        Name = name ?? "";
        Type = typeof(T);
        IsScalar = false;
    }

    public MlxTensor(MlxTensorFactory tf, MlxStream stream, T scalar, string? name = null)
    {
        array = new ManagedMlxArray<T>(scalar);
        this.tf = tf;
        this.stream = stream;
        Name = name ?? "";
        Type = typeof(T);
        IsScalar = true;
    }

    #endregion

    public override TensorShape Shape => array.GetShape();

    #region ObjectMethods

    public override void Release()
    {
        if (!isReleased)
        {
            isReleased = true;
            array.Dispose();
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
        MlxOps.ArrayEqual(in result.Array, array.Array, o.array.Array, false, stream);

        return result.GetScalar();
    }

    public override IEnumerator<T> GetEnumerator()
    {
        return array.GetEnumerator();
    }

    public override object Clone()
    {
        var result = tf.CreateEmpty<T>();
        result.array.CopyFrom(array);
        result.IsScalar = IsScalar;

        return result;
    }

    public override void CopyFrom(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }
        
        array.CopyFrom(o.array);
    }

    public override Tensor<TTarget> ConvertTo<TTarget>()
    {
        var dtype = MlxDType.GetDType<TTarget>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(TTarget));
        }

        var result = tf.CreateEmpty<TTarget>();
        MlxOps.AsType(in result.array.Array, array.Array, dtype.Value, stream);
        result.IsScalar = IsScalar;

        return result;
    }

    public override string ToString()
    {
        return array.ToString();
    }

    internal void SetZero()
    {
        using var shapeHandle = shape?.GetHandle() ?? throw new InvalidOperationException();
        MlxOps.Zeros
        (
            in array.Array,
            (int*)shapeHandle.Pointer,
            (UIntPtr)shape.Dimensions,
            MlxDType.GetDTypeValid<T>(),
            stream
        );
    }

    internal void SetOne()
    {
        using var shapeHandle = shape?.GetHandle() ?? throw new InvalidOperationException();
        MlxOps.Ones
        (
            in array.Array,
            (int*)shapeHandle.Pointer,
            (UIntPtr)shape.Dimensions,
            MlxDType.GetDTypeValid<T>(),
            stream
        );
    }

    #endregion

    #region TensorOps

    #region ArithmeticOps

    public override Tensor<T> Add(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Add(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Sub(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Subtract(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Mul(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Multiply(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> MatMul(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.MatMul(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Div(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Divide(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Mod(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        using var vec = new ManagedMlxVectorArray();
        MlxOps.DivMod(in vec.Vector, array.Array, o.array.Array, stream);
        var res = vec.Get<T>(1);
        
        var result = tf.CreateEmpty<T>();
        result.array.CopyFrom(res);
        res.Dispose();
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Fma(Tensor<T> toMul, Tensor<T> toAdd, float scaleProd = 1.0f, float scaleAdd = 1.0f)
    {
        if (toMul is not MlxTensor<T> mul)
        {
            throw new TensorCompatibilityException();
        }

        if (toAdd is not MlxTensor<T> add)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.AddMM
        (
            in result.array.Array,
            add.array.Array,
            array.Array,
            mul.array.Array,
            scaleProd,
            scaleAdd,
            stream
        );
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Neg()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Negative(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
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
        MlxOps.Equal(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> NotEqual(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.NotEqual(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> Less(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.Less(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> LessEqual(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.LessEqual(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> Greater(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.Greater(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> GreaterEqual(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.GreaterEqual(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> And(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.LogicalAnd(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> Or(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<bool>();
        MlxOps.LogicalOr(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> Not()
    {
        var result = tf.CreateEmpty<bool>();
        MlxOps.LogicalNot(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

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
        MlxOps.BitwiseAnd(in result.array.Array, array.Array, o.array.Array, stream);
        
        return result;
    }

    public override Tensor<T> BitwiseOr(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.BitwiseOr(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> BitwiseXor(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.BitwiseXor(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> BitwiseNot()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.BitwiseInvert(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    #endregion

    #region ExponentialOps

    public override Tensor<T> Exp()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Exp(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Log()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Log(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Log10()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Log10(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Log2()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Log2(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Log1P()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Log1P(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Square()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Square(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Sqrt()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Sqrt(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> RSqrt()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.RSqrt(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Pow(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Power(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    #endregion

    #region TrigonometricOps

    public override Tensor<T> Sin()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Sin(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> SinH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.SinH(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> ArcSin()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcSin(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> ArcSinH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcSinH(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Cos()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Cos(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> CosH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.CosH(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> ArcCos()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcCos(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> ArcCosH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcCosH(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Tan()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Tan(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> TanH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.TanH(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> ArcTan()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcTan(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> ArcTanH()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.ArcTanH(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> ArcTan2(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.ArcTan2(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    #endregion

    #region Rounding

    public override Tensor<T> Floor()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Floor(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Round(int decimals)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Round(in result.array.Array, array.Array, decimals, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Ceil()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Ceil(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Clip(T min, T max)
    {
        using var sMin = new ManagedMlxArray<T>(min);
        using var sMax = new ManagedMlxArray<T>(max);

        var result = tf.CreateEmpty<T>();
        MlxOps.Clip(in result.array.Array, array.Array, sMin.Array, sMax.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    #endregion

    public override Tensor<T> Abs()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Abs(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Sum(bool keepDims)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Sum(in result.array.Array, array.Array, keepDims, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Mean(bool keepDims)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Mean(in result.array.Array, array.Array, keepDims, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Std(bool keepDims, int ddof)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Std(in result.array.Array, array.Array, keepDims, ddof, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Variance(bool keepDims, int ddof)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Var(in result.array.Array, array.Array, keepDims, ddof, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Minimum(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Minimum(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Min(bool keepDims)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Min(in result.array.Array, array.Array, keepDims, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Maximum(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = tf.CreateEmpty<T>();
        MlxOps.Maximum(in result.array.Array, array.Array, o.array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Max(bool keepDims)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Max(in result.array.Array, array.Array, keepDims, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    #region SpecialOps

    public override Tensor<T> Sigmoid()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Sigmoid(in result.array.Array, array.Array, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> Softmax()
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.Softmax(in result.array.Array, array.Array, true, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    public override Tensor<T> TopK(int k)
    {
        var result = tf.CreateEmpty<T>();
        MlxOps.TopK(in result.array.Array, array.Array, k, stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;
        
        return result;
    }

    #endregion

    #endregion
}