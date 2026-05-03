using SonraML.Core.Types;

namespace SonraML.Core.Interfaces;

/// <summary>
/// Interface that all NN modules have to implement.
/// </summary>
/// <typeparam name="T">Type of the tensor for this module.</typeparam>
public interface INNModule<T> where T : struct
{
    public IEnumerable<Parameter<T>> Parameters { get; }
    
    public Tensor<T> Forward(Tensor<T> input);
    
    public Tensor<T> Backward(Tensor<T> gradOutput);

    public Task Save(string filePath);

    public Task Load(string filePath);
}