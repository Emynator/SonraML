using System.Collections;
using SonraML.Core.Exceptions;

namespace SonraML.Core.Types;

public abstract class Tensor<T> : GenericTensor, IEquatable<Tensor<T>>, ICloneable, IEnumerable<T> where T : struct
{
    protected TensorShape? shape;

    #region Properties

    public override TensorShape Shape => shape ?? throw new InvalidOperationException("Shape is not set.");

    public int Size => Shape.Size;

    public int Dimensions => Shape.Dimensions;

    public abstract bool IsScalar { get; }

    #endregion

    #region ObjectMethods

    public virtual bool Equals(Tensor<T>? other)
    {
        if (other is null)
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

        var t = Equal(other);

        return t.All(f => f);
    }

    public override bool Equals(object? obj)
    {
        if (obj is null)
        {
            return false;
        }

        if (ReferenceEquals(this, obj))
        {
            return true;
        }

        if (obj.GetType() != GetType())
        {
            return false;
        }

        return Equals((Tensor<T>)obj);
    }

    public abstract IEnumerator<T> GetEnumerator();

    public override int GetHashCode()
    {
        return HashCode.Combine(Shape.GetHashCode(), Size, IsScalar);
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    public abstract object Clone();

    public Tensor<T> Copy()
    {
        var result = Clone() as Tensor<T>;

        return result ?? throw new BackendOperationException("Expected cloning to return a Tensor<T>.");
    }

    public abstract void CopyFrom(Tensor<T> other);

    public abstract Tensor<TTarget> ConvertTo<TTarget>() where TTarget : struct;

    public virtual void CopyTo(Span<T> destination)
    {
        if (destination.Length < Shape.Size)
        {
            throw new ArgumentOutOfRangeException(nameof(destination), "Destination is too small for this tensor.");
        }

        using var enumerator = GetEnumerator();
        for (var i = 0; i < Shape.Size; i++)
        {
            destination[i] = enumerator.Current;
            enumerator.MoveNext();
        }
    }

    #endregion

    #region TensorOps

    public abstract void EnsureCompute();

    #region ArithmeticOps

    public abstract Tensor<T> Add(Tensor<T> other);

    public abstract Tensor<T> Sub(Tensor<T> other);

    public abstract Tensor<T> Mul(Tensor<T> other);

    public abstract Tensor<T> Rec();

    public abstract Tensor<T> Div(Tensor<T> other);

    public abstract Tensor<T> Mod(Tensor<T> other);

    public abstract Tensor<T> Rem(Tensor<T> other);

    public abstract Tensor<T> Neg();

    public abstract Tensor<T> Abs();

    public abstract Tensor<T> Sign();

    #endregion

    #region LogicalOps

    public abstract Tensor<bool> Equal(Tensor<T> other);

    public abstract Tensor<bool> NotEqual(Tensor<T> other);

    public abstract Tensor<bool> Less(Tensor<T> other);

    public abstract Tensor<bool> LessEqual(Tensor<T> other);

    public abstract Tensor<bool> Greater(Tensor<T> other);

    public abstract Tensor<bool> GreaterEqual(Tensor<T> other);

    public abstract Tensor<bool> And(Tensor<T> other);

    public abstract Tensor<bool> Or(Tensor<T> other);

    public abstract Tensor<bool> Not();

    public abstract Tensor<bool> IsNAN();

    public abstract Tensor<bool> IsInfinity();

    public abstract Tensor<bool> IsFinite();

    public abstract Tensor<bool> IsClose(Tensor<T> other, double rTol, double aTol, bool equalNAN = false);

    public abstract Tensor<bool> AllClose(Tensor<T> other, double rTol, double aTol, bool equalNAN = false);

    #endregion

    #region BitwiseOps

    public abstract Tensor<T> BitwiseAnd(Tensor<T> other);

    public abstract Tensor<T> BitwiseOr(Tensor<T> other);

    public abstract Tensor<T> BitwiseXor(Tensor<T> other);

    public abstract Tensor<T> BitwiseNot();

    #endregion

    #region ExponentialOps

    public abstract Tensor<T> Exp();

    public abstract Tensor<T> ExpM1();

    public abstract Tensor<T> Log();

    public abstract Tensor<T> Log10();

    public abstract Tensor<T> Log2();

    public abstract Tensor<T> Log1P();

    public abstract Tensor<T> Square();

    public abstract Tensor<T> Sqrt();

    public abstract Tensor<T> RSqrt();

    public abstract Tensor<T> Pow(Tensor<T> other);

    public abstract Tensor<T> LogSumExp(bool keepDims = false);

    public abstract Tensor<T> LogSumExp(int axis, bool keepDims = false);

    public abstract Tensor<T> LogSumExp(int[] axes, bool keepDims = false);

    public abstract Tensor<T> LogAddExp(Tensor<T> other);

    #endregion

    #region TrigonometricOps

    public abstract Tensor<T> Sin();

    public abstract Tensor<T> SinH();

    public abstract Tensor<T> ArcSin();

    public abstract Tensor<T> ArcSinH();

    public abstract Tensor<T> Cos();

    public abstract Tensor<T> CosH();

    public abstract Tensor<T> ArcCos();

    public abstract Tensor<T> ArcCosH();

    public abstract Tensor<T> Tan();

    public abstract Tensor<T> TanH();

    public abstract Tensor<T> ArcTan();

    public abstract Tensor<T> ArcTanH();

    public abstract Tensor<T> ArcTan2(Tensor<T> other);

    #endregion

    #region Rounding

    public abstract Tensor<T> Floor();

    public abstract Tensor<T> Round(int decimals);

    public abstract Tensor<T> Ceil();

    public abstract Tensor<T> Clip(T min, T max);

    public abstract Tensor<T> FloorDiv(Tensor<T> other);

    #endregion

    #region MatrixOps

    public abstract Tensor<T> MatMul(Tensor<T> other);

    public abstract Tensor<T> Fma(Tensor<T> b, Tensor<T> c, float alpha = 1.0f, float beta = 1.0f);

    public abstract Tensor<T> Transpose();

    public abstract Tensor<T> Transpose(int[] axes);

    public abstract Tensor<T> SwapAxes(int a, int b);

    public abstract Tensor<T> MoveAxis(int src, int dest);

    public abstract Tensor<T> Diag(int diagonal);

    #endregion

    #region ShapeOps

    public abstract Tensor<T> Reshape(TensorShape shape);

    public abstract Tensor<T> Flatten(int startAxis, int endAxis);

    public abstract Tensor<T> ExpandDims(int axis);

    public abstract Tensor<T> ExpandDims(int[] axes);

    public abstract Tensor<T> BroadcastTo(TensorShape shape);

    #endregion

    #region IndexingOps

    public abstract Tensor<T> Slice(int[] start, int[] stop, int[] strides);

    public abstract Tensor<T> DynamicSlice(Tensor<T> start, int[] axes, int[] sliceSize);

    public abstract Tensor<T> SliceUpdate(Tensor<T> update, int[] start, int[] stop, int[] strides);

    public abstract Tensor<T> Take(Tensor<T> indices);

    public abstract Tensor<T> Take(Tensor<T> indices, int axis);

    public abstract Tensor<T> TakeAlongAxis(Tensor<T> indices, int axis);

    public abstract Tensor<T> Gather(Tensor<T>[] indices, int[] axes, int[] sliceSices);

    #endregion

    #region SplitOps

    public abstract Tensor<T>[] Split(int numSplits, int axis);

    public abstract Tensor<T>[] Split(int[] indices, int axis);

    #endregion

    #region PredicateOps

    public abstract Tensor<T> Sum(bool keepDims = false);

    public abstract Tensor<T> Sum(int axis, bool keepDims = false);

    public abstract Tensor<T> Sum(int[] axes, bool keepDims = false);

    public abstract Tensor<T> Min(bool keepDims = false);

    public abstract Tensor<T> Min(int axis, bool keepDims = false);

    public abstract Tensor<T> Min(int[] axes, bool keepDims = false);

    public abstract Tensor<T> Max(bool keepDims = false);

    public abstract Tensor<T> Max(int axis, bool keepDims = false);

    public abstract Tensor<T> Max(int[] axes, bool keepDims = false);

    public abstract Tensor<T> Mean(bool keepDims = false);

    public abstract Tensor<T> Mean(int axis, bool keepDims = false);

    public abstract Tensor<T> Mean(int[] axes, bool keepDims = false);

    public abstract Tensor<T> Std(int ddof, bool keepDims = false);

    public abstract Tensor<T> Std(int axis, int ddof, bool keepDims = false);

    public abstract Tensor<T> Std(int[] axes, int ddof, bool keepDims = false);

    public abstract Tensor<T> ArgMin(bool keepDims = false);

    public abstract Tensor<T> ArgMin(int axis, bool keepDims = false);

    public abstract Tensor<T> ArgMax(bool keepDims = false);

    public abstract Tensor<T> ArgMax(int axis, bool keepDims = false);

    public abstract Tensor<T> Variance(bool keepDims, int ddof);

    #endregion

    #region SelectionOps

    public abstract Tensor<T> All(bool keepDims = false);

    public abstract Tensor<T> All(int axis, bool keepDims = false);

    public abstract Tensor<T> All(int[] axes, bool keepDims = false);

    public abstract Tensor<T> Any(bool keepDims = false);

    public abstract Tensor<T> Any(int axis, bool keepDims = false);

    public abstract Tensor<T> Any(int[] axes, bool keepDims = false);

    public abstract Tensor<TResult> Where<TResult>
        (
        Tensor<TResult> ifTrue,
        Tensor<TResult> ifFalse
        ) where TResult : struct;

    public abstract Tensor<T> Minimum(Tensor<T> other);

    public abstract Tensor<T> Maximum(Tensor<T> other);

    public abstract Tensor<T> TopK(int k);

    public abstract Tensor<T> TopK(int k, int axis);

    #endregion

    #region LikeOps

    public abstract Tensor<T> ZerosLike();

    public abstract Tensor<T> OnesLike();

    #endregion

    #region NeuralOps

    public abstract Tensor<T> Sigmoid();

    public abstract Tensor<T> Softmax(bool precise = true);

    public abstract Tensor<T> Softmax(int axis, bool precise = true);

    public abstract Tensor<T> Softmax(int[] axes, bool precise = true);

    public abstract Tensor<T> Erf();

    public abstract Tensor<T> ErfInv();

    #endregion

    #endregion

    #region Operators

    public void operator += (Tensor<T> rhs)
    {
        CopyFrom(Add(rhs));
    }

    public void operator -= (Tensor<T> rhs)
    {
        CopyFrom(Sub(rhs));
    }

    public void operator *= (Tensor<T> rhs)
    {
        CopyFrom(Mul(rhs));
    }

    public void operator /= (Tensor<T> rhs)
    {
        CopyFrom(Div(rhs));
    }

    public void operator %= (Tensor<T> rhs)
    {
        Mod(rhs);
    }

    public static Tensor<T> operator +(Tensor<T> lhs, Tensor<T> rhs)
    {
        return lhs.Add(rhs);
    }

    public static Tensor<T> operator -(Tensor<T> lhs, Tensor<T> rhs)
    {
        return lhs.Sub(rhs);
    }

    public static Tensor<T> operator *(Tensor<T> lhs, Tensor<T> rhs)
    {
        return lhs.Mul(rhs);
    }

    public static Tensor<T> operator /(Tensor<T> lhs, Tensor<T> rhs)
    {
        return lhs.Div(rhs);
    }

    public static Tensor<T> operator %(Tensor<T> lhs, Tensor<T> rhs)
    {
        return lhs.Mod(rhs);
    }

    public static bool operator ==(Tensor<T> lhs, Tensor<T> rhs)
    {
        return lhs.Equals(rhs);
    }

    public static bool operator !=(Tensor<T> lhs, Tensor<T> rhs)
    {
        return !lhs.Equals(rhs);
    }

    public static bool operator <(Tensor<T> lhs, Tensor<T> rhs)
    {
        return lhs.Shape.CompareTo(rhs.Shape) < 0;
    }

    public static bool operator >(Tensor<T> lhs, Tensor<T> rhs)
    {
        return lhs.Shape.CompareTo(rhs.Shape) > 0;
    }

    public static bool operator <=(Tensor<T> lhs, Tensor<T> rhs)
    {
        return lhs.Shape.CompareTo(rhs.Shape) <= 0;
    }

    public static bool operator >=(Tensor<T> lhs, Tensor<T> rhs)
    {
        return lhs.Shape.CompareTo(rhs.Shape) >= 0;
    }

    #endregion
}