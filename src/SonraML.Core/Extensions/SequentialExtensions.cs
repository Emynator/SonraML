using SonraML.Core.NN;

namespace SonraML.Core.Extensions;

public static class SequentialExtensions
{
    public static Sequential<T> AddReLU<T>(this Sequential<T> container) where T : struct
    {
        container.Add(new ReLU<T>());

        return container;
    }

    public static Sequential<T> AddSigmoid<T>(this Sequential<T> container) where T : struct
    {
        container.Add(new Sigmoid<T>());
        
        return container;
    }

    public static Sequential<T> AddLinear<T>
        (
        this Sequential<T> container,
        int dimensions,
        int features
        ) where T : struct
    {
        container.Add(new Linear<T>(dimensions, features));
        
        return container;
    }
}