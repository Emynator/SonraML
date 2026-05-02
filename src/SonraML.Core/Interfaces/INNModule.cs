using SonraML.Core.Types;

namespace SonraML.Core.Interfaces;

public interface INNModule<T> where T : struct
{
    public IEnumerable<Parameter<T>> Parameters { get; }
    
    public Tensor<T> Forward(Tensor<T> input);
    
    public Tensor<T> Backward(Tensor<T> gradOutput);
}