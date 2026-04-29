using SonraML.Backend.MLX.Interop;
using SonraML.Core.Enums;

namespace SonraML.Backend.MLX.Managed;

internal class ManagedMlxStream : IDisposable
{
    public ManagedMlxStream(BackendDeviceType type)
    {
        Stream = type switch
        {
            BackendDeviceType.Cpu => MlxStream.DefaultCpuStreamNew(),
            BackendDeviceType.Gpu => MlxStream.DefaultGpuStreamNew(),
            _ => throw new InvalidOperationException(),
        };
    }
    
    public MlxStream Stream { get; init; }
    
    public void Dispose()
    {
        MlxStream.Free(Stream);
    }
}