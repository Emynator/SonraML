using SonraML.Core.Interfaces;

namespace SonraML.Core.Types;

/// <summary>
/// Baseclass that all NN modules have to implement.
/// </summary>
/// <typeparam name="T">Type of the tensor for this module.</typeparam>
public abstract class NNModule<T> where T : struct
{
    public virtual IEnumerable<Parameter<T>> Parameters
    {
        get
        {
            yield break;
        }
    }
    
    public abstract Tensor<T> Forward(Tensor<T> input);
    
    public abstract Tensor<T> Backward(Tensor<T> gradOutput);

    public virtual Task Save(ITensorStore store)
    {
        return Task.CompletedTask;
    }

    public virtual Task Load(ITensorStore store)
    {
        return Task.CompletedTask;
    }
}