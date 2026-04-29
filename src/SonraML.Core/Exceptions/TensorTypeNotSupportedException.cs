namespace SonraML.Core.Exceptions;

public class TensorTypeNotSupportedException : SonraMLBackendException
{
    public TensorTypeNotSupportedException(Type t) : base($"Tensor of type {t.Name} is not supported by backend.")
    {
    }
}