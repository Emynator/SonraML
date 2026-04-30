using SonraML.Backend.MLX.Managed;
using SonraML.Core.Enums;

namespace SonraML.Backend.MLX.Interfaces;

internal interface IMlxBackendGlobals
{
    public BackendDeviceType DeviceType { get; set; }
}