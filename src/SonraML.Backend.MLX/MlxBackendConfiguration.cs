using SonraML.Core;

namespace SonraML.Backend.MLX;

public static class MlxBackendConfiguration
{
    public static void AddMlxBackend()
    {
        SonraMLConfiguration.AddBackend(new MlxBackend());
    }
}