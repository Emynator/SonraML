using SonraML.Backend.MLX.Interfaces;
using SonraML.Core;
using SonraML.Core.Enums;
using SonraML.Core.Interfaces;

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
        globals.DeviceType = type;
    }

    public void Dispose()
    {
    }
}