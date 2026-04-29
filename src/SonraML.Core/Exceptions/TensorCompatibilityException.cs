namespace SonraML.Core.Exceptions;

public class TensorCompatibilityException : SonraMLBackendException
{
    public TensorCompatibilityException() : base("Tensors from different backends are not compatible.")
    {
    }
}