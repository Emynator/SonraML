namespace SonraML.Core.Types;

public abstract class GenericTensor
{
    public string Name { get; init; }

    public Type Type { get; init; }
    
    public virtual TensorShape Shape { get; }

    public Tensor<T> AsTensor<T>() where T : struct
    {
        var result = this as Tensor<T>;
        if (result is null)
        {
            throw new InvalidOperationException($"Can't get Tensor<{Type.Name}> as Tensor<{typeof(T).Name}>.");
        }

        return result;
    }
}