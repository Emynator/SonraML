using SonraML.Core.NN;

namespace SonraML.Core.Extensions;

public static class ContainerExtensions
{
    public static NNContainer<T> AddReLU<T>(this NNContainer<T> container) where T : struct
    {
        container.AddModule(container.Factory.CreateReLU<T>());

        return container;
    }

    public static NNContainer<T> AddSigmoid<T>(this NNContainer<T> container) where T : struct
    {
        container.AddModule(container.Factory.CreateSigmoid<T>());
        
        return container;
    }

    public static NNContainer<T> AddLinear<T>
        (
        this NNContainer<T> container,
        int inputFeatures,
        int outputFeatures,
        string? name = null
        ) where T : struct
    {
        container.AddModule(container.Factory.CreateLinear<T>(inputFeatures, outputFeatures, name));
        
        return container;
    }
}