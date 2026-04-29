using System.Collections;
using SonraML.Core.Exceptions;

namespace SonraML.Core.Types;

public abstract class Tensor<T> : IDisposable, IEquatable<Tensor<T>>, ICloneable, IEnumerable<T> where T : struct
{
    protected TensorShape? shape;
    
    #region StaticCtors

    public static Tensor<T> Zero(TensorShape shape)
    {
        if (!SonraMLConfig.Backend.TensorFactory.IsTypeSupported<T>())
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }

        return SonraMLConfig.Backend.TensorFactory.Zero<T>(shape);
    }

    public static Tensor<T> One(TensorShape shape)
    {
        if (!SonraMLConfig.Backend.TensorFactory.IsTypeSupported<T>())
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }

        return SonraMLConfig.Backend.TensorFactory.One<T>(shape);
    }

    public static Tensor<T> FromMemory(Memory<T> memory, TensorShape shape)
    {
        if (!SonraMLConfig.Backend.TensorFactory.IsTypeSupported<T>())
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }

        return SonraMLConfig.Backend.TensorFactory.Create(memory, shape);
    }

    public static Tensor<T> FromEnumerable(IEnumerable<T> enumerable, TensorShape shape)
    {
        return FromMemory(enumerable.ToArray().AsMemory(), shape);
    }

    public static Tensor<T> FromScalar(T scalar)
    {
        if (!SonraMLConfig.Backend.TensorFactory.IsTypeSupported<T>())
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }

        return SonraMLConfig.Backend.TensorFactory.Create(scalar);
    }

    #endregion

    #region Properties

    public virtual TensorShape Shape => shape ?? throw new InvalidOperationException("Shape is not set."); 

    public int Size => Shape.Size;

    public int Dimensions => Shape.Dimensions;

    public bool IsScalar { get; protected set; }

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

    public abstract void Add(Tensor<T> other);

    public abstract void Sub(Tensor<T> other);

    public abstract void Mul(Tensor<T> other);

    public abstract void Div(Tensor<T> other);
    
    public abstract void Mod(Tensor<T> other);
    
    public abstract void Fma(Tensor<T> toMul, Tensor<T> toAdd, float scaleProd = 1.0f, float scaleAdd = 1.0f);

    public abstract void Neg();

    public abstract Tensor<bool> Equal(Tensor<T> other);

    public abstract Tensor<bool> NotEqual(Tensor<T> other);

    public abstract Tensor<bool> Less(Tensor<T> other);

    public abstract Tensor<bool> LessEqual(Tensor<T> other);

    public abstract Tensor<bool> Greater(Tensor<T> other);

    public abstract Tensor<bool> GreaterEqual(Tensor<T> other);

    public abstract Tensor<bool> And(Tensor<T> other);

    public abstract Tensor<bool> Or(Tensor<T> other);

    public abstract Tensor<bool> Not();

    public abstract void MatMul(Tensor<T> other);

    public abstract void Exp();

    public abstract void Log();

    public abstract void Abs();

    public abstract void Sin();

    public abstract void SinH();

    public abstract void ArcSin();

    public abstract void ArcSinH();

    public abstract void Cos();

    public abstract void CosH();

    public abstract void ArcCos();

    public abstract void ArcCosH();

    public abstract void Tan();

    public abstract void TanH();

    public abstract void ArcTan();

    public abstract void ArcTanH();

    public abstract void ArcTan2(Tensor<T> other);

    public abstract void Clip(T min, T max);

    public abstract void Sum(bool keepDims);

    public abstract void Mean(bool keepDims);

    public abstract void Std(bool keepDims, int ddof);

    public abstract void Variance(bool keepDims, int ddof);

    public abstract void Min(bool keepDims);

    public abstract void Max(bool keepDims);

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