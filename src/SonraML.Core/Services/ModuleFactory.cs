using Microsoft.Extensions.DependencyInjection;
using SonraML.Core.Interfaces;
using SonraML.Core.NN;

namespace SonraML.Core.Services;

public sealed class ModuleFactory
{
    private readonly IServiceProvider serviceProvider;

    public ModuleFactory(IServiceProvider serviceProvider)
    {
        this.serviceProvider = serviceProvider;
    }

    public Sequential<T> CreateSequential<T>() where T : struct
    {
        var module = new Sequential<T>(serviceProvider);
        
        return module;
    }

    public SgdOptimizer<T> CreateSgdOptimizer<T>(T learningRate) where T : struct
    {
        var tf = serviceProvider.GetRequiredService<IScopedTensorFactory>();
        var optimizer = new SgdOptimizer<T>(tf, learningRate);
        
        return optimizer;
    }
}