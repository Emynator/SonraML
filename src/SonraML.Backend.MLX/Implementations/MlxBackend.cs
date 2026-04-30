using SonraML.Backend.MLX.Interfaces;
using SonraML.Core;
using SonraML.Core.Enums;

namespace SonraML.Backend.MLX.Implementations;

internal class MlxBackend : ISonraMLBackend
{
    private readonly IMlxBackendGlobals globals;

    public MlxBackend(IMlxBackendGlobals globals)
    {
        this.globals = globals;
    }

    public void Init(BackendDeviceType type)
    {
        globals.Stream = new(type);
    }

    public void Dispose()
    {
        globals.Stream?.Dispose();
    }
}