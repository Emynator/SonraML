using System.Reflection;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Core.Enums;
using SonraML.Core.Interfaces;
using SonraML.Core.Services;

namespace SonraML.Core;

public static class SonraMLConfiguration
{
    public static IHostApplicationBuilder ConfigureSonraML
        (
        this IHostApplicationBuilder builder,
        Assembly assembly,
        Action<IHostApplicationBuilder> configureBackend
        )
    {
        configureBackend(builder);
        var runners = assembly.GetTypes().Where(t => t.GetInterface(nameof(ISonraRunner)) is not null);
        foreach (var runner in runners)
        {
            builder.Services.AddKeyedScoped(typeof(ISonraRunner), runner.Name, runner);
            builder.Services.AddHostedService<SonraWorker>(services => new SonraWorker(services, runner.Name));
        }

        return builder;
    }

    public static IHost InitSonraML(this IHost host, BackendDeviceType deviceType)
    {
        var backend = host.Services.GetRequiredService<ISonraMLBackend>();
        backend.Init(deviceType);

        return host;
    }
}