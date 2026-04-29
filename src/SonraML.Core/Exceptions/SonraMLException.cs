namespace SonraML.Core.Exceptions;

public abstract class SonraMLException : Exception
{
    public SonraMLException(string message) : base(message)
    {
    }

    public SonraMLException(string message, Exception innerException) : base(message, innerException)
    {
    }
}