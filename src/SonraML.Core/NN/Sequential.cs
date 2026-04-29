using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class Sequential<T> : NNModule<T> where T : struct
{
    private readonly List<NNModule<T>> modules = [];
    
    public override void Forward(Tensor<T> x)
    {
        foreach (var module in modules)
        {
            module.Forward(x);
        }
    }

    public override void Dispose()
    {
        foreach (var module in modules)
        {
            module.Dispose();
        }
    }
    
    internal void Add(NNModule<T> module)
    {
        modules.Add(module);
    }
}