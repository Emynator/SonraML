using SonraML.Core.Enums;

namespace SonraML.Core.Interfaces;

public interface ISonraMLBackend : IDisposable
{
    public void Init(BackendDeviceType type);
}