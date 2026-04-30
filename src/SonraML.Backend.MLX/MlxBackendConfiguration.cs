using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Backend.MLX.Implementations;
using SonraML.Backend.MLX.Interfaces;
using SonraML.Core;
using SonraML.Core.Interfaces;

namespace SonraML.Backend.MLX;

public static class MlxBackendConfiguration
{
    public static void UseMlxBackend(IHostApplicationBuilder builder)
    {
        builder.Services
            .AddSingleton<ISonraMLBackend, MlxBackend>()
            .AddSingleton<IMlxBackendGlobals, MlxBackendGlobals>()
            .AddSingleton<IGlobalTensorFactory, MlxTensorFactory>()
            .AddScoped<IScopedTensorFactory, MlxTensorFactory>();
    }
}