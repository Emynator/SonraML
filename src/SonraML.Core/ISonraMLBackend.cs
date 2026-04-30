using SonraML.Core.Enums;

namespace SonraML.Core;

public interface ISonraMLBackend : IDisposable
{
    public void Init(BackendDeviceType type);
}