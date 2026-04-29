namespace SonraML.Core.Exceptions;

public class BackendNotInitializedException : SonraMLBackendException
{
    public BackendNotInitializedException() : base("Backend is not initialized.")
    {
    }
}