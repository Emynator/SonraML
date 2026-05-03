namespace SonraML.Core.Exceptions;

public sealed class MissingBackendException : SonraMLException
{
    public MissingBackendException() : base("No backend was configured.")
    {
    }
}