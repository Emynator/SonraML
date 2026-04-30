using Microsoft.Extensions.DependencyInjection;
using SonraML.Core.Interfaces;
using SonraML.Core.NN;

namespace SonraML.Core.Extensions;

public static class SequentialExtensions
{
    public static Sequential<T> AddReLU<T>(this Sequential<T> container) where T : struct
    {
        var tf = container.ServiceProvider.GetRequiredService<IScopedTensorFactory>();
        
        container.Add(new ReLU<T>(tf));

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
        int inputFeatures,
        int outputFeatures
        ) where T : struct
    {
        var gtf = container.ServiceProvider.GetRequiredService<IGlobalTensorFactory>();
        var tf = container.ServiceProvider.GetRequiredService<IScopedTensorFactory>();
        
        container.Add(new Linear<T>(gtf, tf, inputFeatures, outputFeatures));
        
        return container;
    }
}