using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using SonraML.Core.Interfaces;
using SonraML.Core.NN;

namespace SonraML.Core.Extensions;

public static class ContainerExtensions
{
    public static NNContainer<T> AddReLU<T>(this NNContainer<T> container) where T : struct
    {
        var tf = container.ServiceProvider.GetRequiredService<IGlobalTensorFactory>();
        
        container.AddModule(new ReLU<T>(tf));

        return container;
    }

    public static NNContainer<T> AddSigmoid<T>(this NNContainer<T> container) where T : struct
    {
        container.AddModule(new Sigmoid<T>());
        
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
        var gtf = container.ServiceProvider.GetRequiredService<IGlobalTensorFactory>();
        var logger = container.ServiceProvider.GetRequiredService<ILogger<Linear<T>>>();
        
        container.AddModule(new Linear<T>(logger, gtf, inputFeatures, outputFeatures, name));
        
        return container;
    }
}