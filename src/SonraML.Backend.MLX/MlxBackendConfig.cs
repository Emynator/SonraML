using SonraML.Core;

namespace SonraML.Backend.MLX;

public static class MlxBackendConfig
{
    public static void AddMlxBackend()
    {
        SonraMLConfig.AddBackend(new MlxBackend());
    }
}