using SonraML.Core.NN;

namespace SonraML.Core.Services;

public class ModuleFactory
{
    private readonly IServiceProvider serviceProvider;

    ModuleFactory(IServiceProvider serviceProvider)
    {
        this.serviceProvider = serviceProvider;
    }

    public Sequential<T> CreateSequential<T>() where T : struct
    {
        var module = new Sequential<T>(serviceProvider);
        
        return module;
    }
}