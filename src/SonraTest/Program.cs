using SonraML.Backend.MLX;
using SonraML.Core;
using SonraML.Core.Enums;

namespace SonraTest;

public static class Program
{
    public static void Main(string[] args)
    {
        var builder = Host.CreateApplicationBuilder(args);
        builder.ConfigureSonraML(MlxBackendConfiguration.UseMlxBackend);

        var host = builder.Build();
        host.InitSonraML(BackendDeviceType.Gpu);
        host.Run();
    }
}