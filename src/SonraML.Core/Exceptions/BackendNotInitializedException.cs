namespace SonraML.Core.Exceptions;

public sealed class BackendNotInitializedException : SonraMLBackendException
{
    public BackendNotInitializedException() : base("Backend is not initialized.")
    {
    }
}