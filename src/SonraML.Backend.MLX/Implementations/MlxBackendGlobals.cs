using SonraML.Backend.MLX.Interfaces;
using SonraML.Backend.MLX.Managed;
using SonraML.Core.Enums;

namespace SonraML.Backend.MLX.Implementations;

internal class MlxBackendGlobals : IMlxBackendGlobals
{
    public BackendDeviceType DeviceType { get; set; } = BackendDeviceType.None;
}