using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Core.Config;
using SonraML.Core.Enums;
using SonraML.Core.Interfaces;
using SonraML.Core.Services;
using SonraML.Core.Types;

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
        builder.Services.AddScoped<ModuleFactory>();
        builder.Services
            .AddOptions<SonraRunnerConfigurations>()
            .BindConfiguration(nameof(SonraRunnerConfigurations));

        return builder;
    }

    public static IServiceCollection AddRunner<TRunner, TContext>
        (this IServiceCollection services) where TRunner : SonraRunner where TContext : ISonraRunnerContext
    {
        var rname = typeof(TRunner).Name;
        var cname = $"{typeof(TContext).Name}_Context";
        services.AddKeyedScoped(typeof(SonraRunner), rname, typeof(TRunner));
        services.AddKeyedScoped(typeof(ISonraRunnerContext), cname, typeof(TContext));
        services.AddHostedService<SonraWorker>(sp => new SonraWorker(sp, rname, cname));

        return services;
    }

    public static IHost InitSonraML(this IHost host, BackendDeviceType deviceType)
    {
        var backend = host.Services.GetRequiredService<ISonraBackend>();
        backend.Init(deviceType);

        return host;
    }
}