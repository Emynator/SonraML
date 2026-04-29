namespace SonraML.Core.Types;

public class TensorShape : IEquatable<TensorShape>, IComparable<TensorShape>
{
    private readonly int[] shape;

    public TensorShape(int[] shape)
    {
        this.shape = shape;
    }
    
    public int Size => shape.Sum();
    
    public int Dimensions => shape.Length;

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
        return shape.GetHashCode();
    }
}