namespace SonraML.Core.Types;

public class TensorShape : IEquatable<TensorShape>, IComparable<TensorShape>
{
    public TensorShape(int[] shape)
    {
        Shape = shape;
    }
    
    public int[] Shape { get; init; }
    
    public int Size => Shape.Sum();
    
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