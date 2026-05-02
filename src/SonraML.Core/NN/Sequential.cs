using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class Sequential<T> : INNModule<T> where T : struct
{
    private readonly List<INNModule<T>> modules = [];

    public Sequential(IServiceProvider serviceProvider)
    {
        ServiceProvider = serviceProvider;
    }
    
    public IServiceProvider ServiceProvider { get; }

    public IEnumerable<Parameter<T>> Parameters => modules.SelectMany(module => module.Parameters);

    public Tensor<T> Forward(Tensor<T> input)
    {
        var result = input.Copy();
        foreach (var module in modules)
        {
            result = module.Forward(result);
        }
        
        return result;
    }

    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var result = gradOutput.Copy();
        for (var i = modules.Count - 1; i > 0; i--)
        {
            result = modules[i].Backward(result);
        }
        
        return result;
    }

    public void AddModule(INNModule<T> module)
    {
        modules.Add(module);
    }
}