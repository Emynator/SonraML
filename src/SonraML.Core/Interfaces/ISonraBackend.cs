using SonraML.Core.Enums;

namespace SonraML.Core.Interfaces;

public interface ISonraBackend : IDisposable
{
    public void Init(BackendDeviceType type);
}