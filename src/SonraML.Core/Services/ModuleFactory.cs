namespace SonraML.Core.Services;

public sealed class ModuleFactory
{
    public ModuleFactory(IServiceProvider serviceProvider)
    {
        ServiceProvider = serviceProvider;
    }
    
    public IServiceProvider ServiceProvider { get; }
}