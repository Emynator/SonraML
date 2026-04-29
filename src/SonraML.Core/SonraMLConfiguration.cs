using SonraML.Core.Enums;
using SonraML.Core.Exceptions;

namespace SonraML.Core;

public static class SonraMLConfiguration
{
    private static SonraMLBackend? backend;

    public static SonraMLBackend Backend => backend ?? throw new BackendNotInitializedException();

    public static void AddBackend(SonraMLBackend b)
    {
        backend = b;
    }

    public static void Init(BackendDeviceType type)
    {
        if (backend is null)
        {
            throw new MissingBackendException();
        }
        
        backend.Init(type);
    }

    public static void Shutdown()
    {
        backend?.Dispose();
        backend = null;
    }
}