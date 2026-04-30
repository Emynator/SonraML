using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class Sequential<T> : NNModule<T> where T : struct
{
    private readonly List<NNModule<T>> modules = [];

    public Sequential(IServiceProvider serviceProvider)
    {
        ServiceProvider = serviceProvider;
    }
    
    internal IServiceProvider ServiceProvider { get; init; }
    
    public override Tensor<T> Forward(Tensor<T> x)
    {
        var result = x.Copy();
        foreach (var module in modules)
        {
            x = module.Forward(x);
        }
        
        return result;
    }
    
    internal void Add(NNModule<T> module)
    {
        modules.Add(module);
    }
}