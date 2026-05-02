using System.Reflection;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Core.Enums;
using SonraML.Core.Interfaces;
using SonraML.Core.Services;

namespace SonraML.Core;

public static class SonraMLConfiguration
{
    internal static Dictionary<string, string> RunnerConfig = [];

    public static IHostApplicationBuilder ConfigureSonraML
        (
        this IHostApplicationBuilder builder,
        Action<IHostApplicationBuilder> configureBackend
        )
    {
        configureBackend(builder);

        return builder;
    }

    public static IServiceCollection AddRunner<TRunner, TContext>
        (this IServiceCollection services) where TRunner : ISonraRunner where TContext : ISonraRunnerContext
    {
        var rname = typeof(TRunner).Name;
        var cname = $"{typeof(TContext).Name}_Context";
        if (RunnerConfig.TryAdd(rname, cname))
        {
            services.AddKeyedScoped(typeof(ISonraRunner), rname, typeof(TRunner));
            services.AddKeyedScoped(typeof(ISonraRunnerContext), cname, typeof(TContext));
            services.AddHostedService<SonraWorker>
            (sp =>
                new SonraWorker(sp, rname, cname)
            );
        }

        return services;
    }

    public static IHost InitSonraML(this IHost host, BackendDeviceType deviceType)
    {
        var backend = host.Services.GetRequiredService<ISonraMLBackend>();
        backend.Init(deviceType);

        return host;
    }
}