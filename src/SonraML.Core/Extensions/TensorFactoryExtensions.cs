using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.Extensions;

public static class TensorFactoryExtensions
{
    public static Tensor<T> FromArray<T>
        (
        this ITensorFactory factory,
        T[] array,
        TensorShape shape,
        string? name = null
        ) where T : struct
    {
        return factory.Create(array.AsMemory(), shape, name);
    }
    
    public static Tensor<T> FromEnumerable<T>
        (
        this ITensorFactory factory,
        IEnumerable<T> enumerable,
        TensorShape shape,
        string? name = null
        ) where T : struct
    {
        return factory.Create(enumerable.ToArray().AsMemory(), shape, name);
    }
}