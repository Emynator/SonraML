using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public abstract class NNContainer<T> : INNModule<T> where T : struct
{
    public NNContainer(IServiceProvider serviceProvider)
    {
        ServiceProvider = serviceProvider;
    }
    
    public IServiceProvider ServiceProvider { get; }
    
    public abstract IEnumerable<Parameter<T>> Parameters { get; }

    public abstract Tensor<T> Forward(Tensor<T> input);

    public abstract Tensor<T> Backward(Tensor<T> gradOutput);

    public abstract Task Save(ITensorStore store);

    public abstract Task Load(ITensorStore store);

    public abstract void AddModule(INNModule<T> module);
}