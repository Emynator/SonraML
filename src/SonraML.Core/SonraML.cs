using SonraML.Core.Exceptions;

namespace SonraML.Core;

public static class SonraML
{
    private static SonraMLBackend? backend;

    public static SonraMLBackend Backend => backend ?? throw new BackendNotInitializedException();

    public static void AddBackend(SonraMLBackend b)
    {
        backend = b;
    }

    public static void Shutdown()
    {
        backend?.Dispose();
        backend = null;
    }
}