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

    public bool IsScalar { get; protected set; }

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

    #region ArithmeticOps
    
    public abstract Tensor<T> Add(Tensor<T> other);

    public abstract Tensor<T> Sub(Tensor<T> other);

    public abstract Tensor<T> Mul(Tensor<T> other);
    
    public abstract Tensor<T> MatMul(Tensor<T> other);

    public abstract Tensor<T> Div(Tensor<T> other);
    
    public abstract Tensor<T> Mod(Tensor<T> other);
    
    public abstract Tensor<T> Fma(Tensor<T> toMul, Tensor<T> toAdd, float scaleProd = 1.0f, float scaleAdd = 1.0f);

    public abstract Tensor<T> Neg();
    
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
    
    #endregion

    #region BitwiseOps

    public abstract Tensor<T> BitwiseAnd(Tensor<T> other);
    
    public abstract Tensor<T> BitwiseOr(Tensor<T> other);
    
    public abstract Tensor<T> BitwiseXor(Tensor<T> other);

    public abstract Tensor<T> BitwiseNot();

    #endregion
    
    #region ExponentialOps

    public abstract Tensor<T> Exp();

    public abstract Tensor<T> Log();

    public abstract Tensor<T> Log10();

    public abstract Tensor<T> Log2();

    public abstract Tensor<T> Log1P();

    public abstract Tensor<T> Square();

    public abstract Tensor<T> Sqrt();

    public abstract Tensor<T> RSqrt();
    
    public abstract Tensor<T> Pow(Tensor<T> other);
    
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

    #endregion

    public abstract Tensor<T> Abs();

    public abstract Tensor<T> Sum(bool keepDims);

    public abstract Tensor<T> Mean(bool keepDims);

    public abstract Tensor<T> Std(bool keepDims, int ddof);

    public abstract Tensor<T> Variance(bool keepDims, int ddof);
    
    public abstract Tensor<T> Minimum(Tensor<T> other);

    public abstract Tensor<T> Min(bool keepDims);
    
    public abstract Tensor<T> Maximum(Tensor<T> other);

    public abstract Tensor<T> Max(bool keepDims);

    #region SpecialOps

    public abstract Tensor<T> Sigmoid();

    public abstract Tensor<T> Softmax();

    public abstract Tensor<T> TopK(int k);

    #endregion

    #endregion

    #region Operators

    public void operator += (Tensor<T> rhs)
    {
        Add(rhs);
    }

    public void operator -= (Tensor<T> rhs)
    {
        Sub(rhs);
    }

    public void operator *= (Tensor<T> rhs)
    {
        Mul(rhs);
    }

    public void operator /= (Tensor<T> rhs)
    {
        Div(rhs);
    }

    public void operator %= (Tensor<T> rhs)
    {
        Mod(rhs);
    }

    public static Tensor<T> operator +(Tensor<T> lhs, Tensor<T> rhs)
    {
        var result = lhs.Copy();
        result.Add(rhs);

        return result;
    }

    public static Tensor<T> operator -(Tensor<T> lhs, Tensor<T> rhs)
    {
        var result = lhs.Copy();
        result.Sub(rhs);

        return result;
    }

    public static Tensor<T> operator *(Tensor<T> lhs, Tensor<T> rhs)
    {
        var result = lhs.Copy();
        result.Mul(rhs);

        return result;
    }

    public static Tensor<T> operator /(Tensor<T> lhs, Tensor<T> rhs)
    {
        var result = lhs.Copy();
        result.Div(rhs);

        return result;
    }

    public static Tensor<T> operator %(Tensor<T> lhs, Tensor<T> rhs)
    {
        var result = lhs.Copy();
        result.Mod(rhs);
        
        return result;
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