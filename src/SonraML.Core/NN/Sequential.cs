using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public sealed class Sequential<T> : NNContainer<T> where T : struct
{
    private readonly List<INNModule<T>> modules = [];

    public Sequential(IServiceProvider serviceProvider) : base(serviceProvider)
    {
    }
    
    public override IEnumerable<Parameter<T>> Parameters => modules.SelectMany(module => module.Parameters);

    public override Tensor<T> Forward(Tensor<T> input)
    {
        var result = input.Copy();
        foreach (var module in modules)
        {
            result = module.Forward(result);
        }
        
        return result;
    }

    public override Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var result = gradOutput.Copy();
        for (var i = modules.Count - 1; i > 0; i--)
        {
            result = modules[i].Backward(result);
        }
        
        return result;
    }

    public override async Task Save(string filePath)
    {
    }

    public override async Task Load(string filePath)
    {
    }

    public override void AddModule(INNModule<T> module)
    {
        modules.Add(module);
    }
}