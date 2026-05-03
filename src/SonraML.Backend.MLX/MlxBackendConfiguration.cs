using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Backend.MLX.ExecutionManagement;
using SonraML.Backend.MLX.Implementations;
using SonraML.Core.Interfaces;

namespace SonraML.Backend.MLX;

public static class MlxBackendConfiguration
{
    public static void UseMlxBackend(IHostApplicationBuilder builder)
    {
        builder.Services
            .AddSingleton<ISonraBackend, MlxBackend>()
            .AddSingleton<MlxBackendGlobals>()
            .AddSingleton<IGlobalTensorFactory, MlxTensorFactory>()
            .AddSingleton<MlxScheduler>()
            .AddSingleton<MlxTensorManager>()
            .AddScoped<IScopedTensorFactory, MlxTensorFactory>();
    }
}