namespace SonraML.Core.Exceptions;

public class BackendOperationException : SonraMLBackendException
{
    public BackendOperationException(string message) : base(message)
    {
    }

    public BackendOperationException(string message, Exception innerException) : base(message, innerException)
    {
    }
}