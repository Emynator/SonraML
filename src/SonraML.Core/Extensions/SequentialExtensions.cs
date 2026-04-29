using SonraML.Core.NN;

namespace SonraML.Core.Extensions;

public static class SequentialExtensions
{
    public static Sequential<T> AddReLU<T>(this Sequential<T> container) where T : struct
    {
        container.Append(new ReLU<T>());

        return container;
    }

    public static Sequential<T> AddSigmoid<T>(this Sequential<T> container) where T : struct
    {
        container.Append(new Sigmoid<T>());
        
        return container;
    }

    public static Sequential<T> AddLinear<T>
        (
        this Sequential<T> container,
        int featuresIn,
        int featuresOut
        ) where T : struct
    {
        container.Append(new Linear<T>(featuresIn, featuresOut));
        
        return container;
    }
}