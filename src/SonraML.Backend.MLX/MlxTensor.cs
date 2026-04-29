using SonraML.Backend.MLX.Extensions;
using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Backend.MLX.Managed;
using SonraML.Core.Exceptions;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX;

internal unsafe class MlxTensor<T> : Tensor<T> where T : struct
{
    private readonly ManagedMlxArray<T> array;

    #region Ctors

    private MlxTensor()
    {
        array = new ManagedMlxArray<T>();
    }

    internal MlxTensor(TensorShape shape)
    {
        array = new ManagedMlxArray<T>();
        this.shape = shape;
    }

    internal MlxTensor(Memory<T> array, TensorShape shape)
    {
        this.array = new ManagedMlxArray<T>(array, shape);
        IsScalar = false;
    }

    internal MlxTensor(T scalar)
    {
        array = new ManagedMlxArray<T>(scalar);
        IsScalar = true;
    }

    #endregion

    public override TensorShape Shape => array.GetShape();

    private MlxStream Stream => MlxBackend.Instance.Stream.Stream;

    #region ObjectMethods

    public override void Dispose()
    {
        array.Dispose();
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
        MlxOps.ArrayEqual(in result.Array, array.Array, o.array.Array, false, Stream);
        
        return result.GetScalar();
    }

    public override IEnumerator<T> GetEnumerator()
    {
        return array.GetEnumerator();
    }

    public override object Clone()
    {
        var result = new MlxTensor<T>();
        result.array.CopyFrom(array);
        result.IsScalar = IsScalar;

        return result;
    }

    public override Tensor<TTarget> ConvertTo<TTarget>()
    {
        var dtype = MlxDType.GetDType<TTarget>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(TTarget));
        }

        var result = new MlxTensor<TTarget>();
        MlxOps.AsType(in result.array.Array, array.Array, dtype.Value, Stream);
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
            Stream
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
            Stream
        );
    }

    #endregion

    #region TensorOps

    public override void Add(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        MlxOps.Add(in array.Array, array.Array, o.array.Array, Stream);
    }

    public override void Sub(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        MlxOps.Subtract(in array.Array, array.Array, o.array.Array, Stream);
    }

    public override void Mul(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        MlxOps.Multiply(in array.Array, array.Array, o.array.Array, Stream);
    }

    public override void Div(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        MlxOps.Divide(in array.Array, array.Array, o.array.Array, Stream);
    }

    public override void Mod(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        using var vec = new ManagedMlxVectorArray();
        MlxOps.DivMod(in vec.Vector, array.Array, o.array.Array, Stream);
        var res = vec.Get<T>(1);
        array.CopyFrom(res);
        res.Dispose();
    }
    
    public override void Fma(Tensor<T> toMul, Tensor<T> toAdd, float scaleProd = 1.0f, float scaleAdd = 1.0f)
    {
        if (toMul is not MlxTensor<T> mul)
        {
            throw new TensorCompatibilityException();
        }

        if (toAdd is not MlxTensor<T> add)
        {
            throw new TensorCompatibilityException();
        }

        MlxOps.AddMM
        (
            in array.Array,
            add.array.Array,
            array.Array,
            mul.array.Array,
            scaleProd,
            scaleAdd,
            Stream
        );
    }

    public override void Neg()
    {
        MlxOps.Negative(in array.Array, array.Array, Stream);
    }

    public override Tensor<bool> Equal(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = new MlxTensor<bool>();
        MlxOps.Equal(in result.array.Array, array.Array, o.array.Array, Stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> NotEqual(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = new MlxTensor<bool>();
        MlxOps.NotEqual(in result.array.Array, array.Array, o.array.Array, Stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> Less(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = new MlxTensor<bool>();
        MlxOps.Less(in result.array.Array, array.Array, o.array.Array, Stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> LessEqual(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = new MlxTensor<bool>();
        MlxOps.LessEqual(in result.array.Array, array.Array, o.array.Array, Stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> Greater(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = new MlxTensor<bool>();
        MlxOps.Greater(in result.array.Array, array.Array, o.array.Array, Stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> GreaterEqual(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = new MlxTensor<bool>();
        MlxOps.GreaterEqual(in result.array.Array, array.Array, o.array.Array, Stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> And(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = new MlxTensor<bool>();
        MlxOps.LogicalAnd(in result.array.Array, array.Array, o.array.Array, Stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> Or(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        var result = new MlxTensor<bool>();
        MlxOps.LogicalOr(in result.array.Array, array.Array, o.array.Array, Stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override Tensor<bool> Not()
    {
        var result = new MlxTensor<bool>();
        MlxOps.LogicalNot(in result.array.Array, array.Array, Stream);
        result.IsScalar = MlxArray.NDim(result.array.Array) == 0;

        return result;
    }

    public override void MatMul(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        MlxOps.MatMul(in array.Array, array.Array, o.array.Array, Stream);
    }

    public override void Exp()
    {
        MlxOps.Exp(in array.Array, array.Array, Stream);
    }

    public override void Log()
    {
        MlxOps.Log(in array.Array, array.Array, Stream);
    }

    public override void Abs()
    {
        MlxOps.Abs(in array.Array, array.Array, Stream);
    }

    public override void Sin()
    {
        MlxOps.Sin(in array.Array, array.Array, Stream);
    }

    public override void SinH()
    {
        MlxOps.SinH(in array.Array, array.Array, Stream);
    }

    public override void ArcSin()
    {
        MlxOps.ArcSin(in array.Array, array.Array, Stream);
    }

    public override void ArcSinH()
    {
        MlxOps.ArcSinH(in array.Array, array.Array, Stream);
    }

    public override void Cos()
    {
        MlxOps.Cos(in array.Array, array.Array, Stream);
    }

    public override void CosH()
    {
        MlxOps.CosH(in array.Array, array.Array, Stream);
    }

    public override void ArcCos()
    {
        MlxOps.ArcCos(in array.Array, array.Array, Stream);
    }

    public override void ArcCosH()
    {
        MlxOps.ArcCosH(in array.Array, array.Array, Stream);
    }

    public override void Tan()
    {
        MlxOps.Tan(in array.Array, array.Array, Stream);
    }

    public override void TanH()
    {
        MlxOps.TanH(in array.Array, array.Array, Stream);
    }

    public override void ArcTan()
    {
        MlxOps.ArcTan(in array.Array, array.Array, Stream);
    }

    public override void ArcTanH()
    {
        MlxOps.ArcTanH(in array.Array, array.Array, Stream);
    }

    public override void ArcTan2(Tensor<T> other)
    {
        if (other is not MlxTensor<T> o)
        {
            throw new TensorCompatibilityException();
        }

        MlxOps.ArcTan2(in array.Array, array.Array, o.array.Array, Stream);
    }

    public override void Clip(T min, T max)
    {
        using var sMin = new ManagedMlxArray<T>(min);
        using var sMax = new ManagedMlxArray<T>(max);

        MlxOps.Clip(in array.Array, array.Array, sMin.Array, sMax.Array, Stream);
    }

    public override void Sum(bool keepDims)
    {
        MlxOps.Sum(in array.Array, array.Array, keepDims, Stream);
    }

    public override void Mean(bool keepDims)
    {
        MlxOps.Mean(in array.Array, array.Array, keepDims, Stream);
    }

    public override void Std(bool keepDims, int ddof)
    {
        MlxOps.Std(in array.Array, array.Array, keepDims, ddof, Stream);
    }

    public override void Variance(bool keepDims, int ddof)
    {
        MlxOps.Var(in array.Array, array.Array, keepDims, ddof, Stream);
    }

    public override void Min(bool keepDims)
    {
        MlxOps.Min(in array.Array, array.Array, keepDims, Stream);
    }

    public override void Max(bool keepDims)
    {
        MlxOps.Max(in array.Array, array.Array, keepDims, Stream);
    }

    #endregion
}