using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using SonraML.Core.Interfaces;
using SonraML.Core.NN;
using SonraML.Core.Services;

namespace SonraML.Core.Extensions;

public static class ModuleFactoryExtensions
{
    public static Sequential<T> CreateSequential<T>(this ModuleFactory factory) where T : struct
    {
        return new Sequential<T>(factory);
    }

    public static Linear<T> CreateLinear<T>
        (
        this ModuleFactory factory,
        int inputFeatures,
        int outputFeatures,
        string? name = null
        ) where T : struct
    {
        var logger = factory.ServiceProvider.GetRequiredService<ILogger<Linear<T>>>();
        var tf = factory.ServiceProvider.GetRequiredService<IGlobalTensorFactory>();
        
        return new Linear<T>(logger, tf, inputFeatures, outputFeatures, name);
    }

    public static ReLU<T> CreateReLU<T>(this ModuleFactory factory) where T : struct
    {
        var tf = factory.ServiceProvider.GetRequiredService<IGlobalTensorFactory>();

        return new ReLU<T>(tf);
    }

    public static Sigmoid<T> CreateSigmoid<T>(this ModuleFactory factory) where T : struct
    {
        return new Sigmoid<T>();
    }

    public static SgdOptimizer<T> CreateSgdOptimizer<T>(this ModuleFactory factory, T learningRate) where T : struct
    {
        var tf = factory.ServiceProvider.GetRequiredService<IScopedTensorFactory>();
        return new SgdOptimizer<T>(tf, learningRate);
    }
}