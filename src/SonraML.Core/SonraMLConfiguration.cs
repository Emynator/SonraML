using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Core.Enums;

namespace SonraML.Core;

public static class SonraMLConfiguration
{
    public static IHostApplicationBuilder ConfigureSonraML
        (
        this IHostApplicationBuilder builder,
        Action<IHostApplicationBuilder> configureBackend
        )
    {
        configureBackend(builder);
        
        return builder;
    }

    public static IHost InitSonraML(this IHost host, BackendDeviceType deviceType)
    {
        var backend = host.Services.GetRequiredService<ISonraMLBackend>();
        backend.Init(deviceType);
        
        return host;
    }
}