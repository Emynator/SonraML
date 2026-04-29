using System.Collections;
using SonraML.Core.Exceptions;

namespace SonraML.Core.Types;

public abstract class Tensor<T> : IDisposable, IEquatable<Tensor<T>>, ICloneable, IEnumerable<T> where T : struct
{
    #region StaticCtors

    public static Tensor<T> Zero(TensorShape shape)
    {
        if (!SonraML.Backend.TensorFactory.IsTypeSupported(typeof(T)))
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }

        return SonraML.Backend.TensorFactory.Zero<T>(shape);
    }

    public static Tensor<T> One(TensorShape shape)
    {
        if (!SonraML.Backend.TensorFactory.IsTypeSupported(typeof(T)))
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }

        return SonraML.Backend.TensorFactory.One<T>(shape);
    }

    public static Tensor<T> FromSpan(Span<T> span, TensorShape shape)
    {
        if (!SonraML.Backend.TensorFactory.IsTypeSupported(typeof(T)))
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        return SonraML.Backend.TensorFactory.Create(span, shape);
    }

    public static Tensor<T> FromEnumerable(IEnumerable<T> enumerable, TensorShape shape)
    {
        return FromSpan(enumerable.ToArray().AsSpan(), shape); 
    }

    public static Tensor<T> FromScalar(T scalar)
    {
        if (!SonraML.Backend.TensorFactory.IsTypeSupported(typeof(T)))
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }

        return SonraML.Backend.TensorFactory.Create(scalar);
    }

    #endregion

    #region Properties

    public abstract TensorShape Shape { get; protected set; }
    
    public int Size => Shape.Size;
    
    public int Dimensions => Shape.Dimensions;

    public abstract bool IsScalar { get; protected set; }

    #endregion

    #region ObjectMethods

    public abstract void Dispose();

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

        using var t = Equal(other);
        
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

    public abstract void CopyTo(Span<T> destination);

    #endregion

    #region TensorOps

    public abstract void Add(Tensor<T> other);

    public abstract void Sub(Tensor<T> other);

    public abstract void Mul(Tensor<T> other);

    public abstract void Div(Tensor<T> other);

    public abstract void Fma(Tensor<T> toAdd, Tensor<T> toMul);

    public abstract void Neg();

    public abstract Tensor<bool> Equal(Tensor<T> other);

    public abstract Tensor<bool> NotEqual(Tensor<T> other);

    public abstract Tensor<bool> Less(Tensor<T> other);

    public abstract Tensor<bool> LessEqual(Tensor<T> other);

    public abstract Tensor<bool> Greater(Tensor<T> other);

    public abstract Tensor<bool> GreaterEqual(Tensor<T> other);

    public abstract Tensor<bool> And(Tensor<T> other);

    public abstract Tensor<bool> Or(Tensor<T> other);

    public abstract void MatAdd(Tensor<T> other);

    public abstract void MatSub(Tensor<T> other);

    public abstract void MatMul(Tensor<T> other);

    public abstract void MatDiv(Tensor<T> other);

    public abstract void MatFma(Tensor<T> toMul, Tensor<T> toAdd);

    public abstract void Not();

    public abstract void Exp();

    public abstract void Log();

    public abstract void Abs();

    public abstract void Sin();

    public abstract void Sinh();

    public abstract void ArcSin();

    public abstract void ArcSinH();

    public abstract void Cos();

    public abstract void Cosh();

    public abstract void ArcCos();

    public abstract void ArcCosH();

    public abstract void Tan();

    public abstract void Tanh();

    public abstract void ArcTan();

    public abstract void ArcTan2();

    public abstract void ArcTanH();

    public abstract void Clip(T min, T max);

    public abstract void Sum();

    public abstract void Mean();

    public abstract void Std();

    public abstract void Variance();

    public abstract void Min();

    public abstract void Max();

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

    public static bool operator ==(Tensor<T> lhs, Tensor<T> rhs)
    {
        return lhs.Equals(rhs);
    }

    public static bool operator !=(Tensor<T> lhs, Tensor<T> rhs)
    {
        return !lhs.Equals(rhs);
    }

    #endregion
}