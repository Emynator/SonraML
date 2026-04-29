namespace SonraML.Core.Exceptions;

public class MissingBackendException : SonraMLException
{
    public MissingBackendException() : base("No backend was configured.")
    {
    }
}