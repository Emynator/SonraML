namespace SonraML.Core.Types;

public sealed class TensorShape : IEquatable<TensorShape>, IComparable<TensorShape>
{
    public TensorShape(int[] shape)
    {
        Shape = shape;
    }
    
    public int[] Shape { get; }
    
    public int Size => Shape.Aggregate(1, (size, dim) => size * dim);
    
    public int Dimensions => Shape.Length;

    public bool Equals(TensorShape? other)
    {
        if (other is null)
        {
            return false;
        }

        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return CompareTo(other) == 0;
    }

    public int CompareTo(TensorShape? other)
    {
        if (other is null)
        {
            return 1;
        }
        
        return Size - other.Size;
    }
    
    public override int GetHashCode()
    {
        return Shape.GetHashCode();
    }
}