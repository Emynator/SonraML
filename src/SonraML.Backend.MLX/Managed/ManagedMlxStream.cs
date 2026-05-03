using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Core.Enums;

namespace SonraML.Backend.MLX.Managed;

internal class ManagedMlxStream : IDisposable
{
    public ManagedMlxStream(MlxDeviceType type)
    {
        Stream = type switch
        {
            MlxDeviceType.Cpu => MlxStream.DefaultCpuStreamNew(),
            MlxDeviceType.Gpu => MlxStream.DefaultGpuStreamNew(),
            _ => throw new InvalidOperationException(),
        };
    }
    
    public MlxStream Stream { get; init; }
    
    public void Dispose()
    {
        MlxStream.Free(Stream);
    }
}