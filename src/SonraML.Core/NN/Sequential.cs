using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class Sequential<T> : NNModule<T> where T : struct
{
    private readonly List<NNModule<T>> modules = [];
    
    public override Tensor<T> Forward(Tensor<T> x)
    {
        foreach (var module in modules)
        {
            x = module.Forward(x);
        }
        
        return x;
    }

    internal void Append(NNModule<T> module)
    {
        modules.Add(module);
    }
}