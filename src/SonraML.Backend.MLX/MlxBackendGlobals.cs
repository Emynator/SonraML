using System.Collections.Concurrent;
using SonraML.Backend.MLX.ExecutionManagement;
using SonraML.Core.Enums;

namespace SonraML.Backend.MLX;

internal class MlxBackendGlobals
{
    public BackendDeviceType DeviceType { get; set; } = BackendDeviceType.None;
    
    public ConcurrentQueue<MlxCommand> CommandQueue { get; } = new();
}