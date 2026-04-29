namespace SonraML.Core.Exceptions;

public abstract class SonraMLBackendException : SonraMLException
{
    public SonraMLBackendException(string message) : base(message)
    {
    }

    public SonraMLBackendException(string message, Exception innerException) : base(message, innerException)
    {
    }
}