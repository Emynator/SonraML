using SonraML.Core.Interfaces;
using SonraML.Core.Services;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public sealed class Sequential<T> : NNContainer<T> where T : struct
{
    private readonly List<NNModule<T>> modules = [];

    public Sequential(ModuleFactory factory) : base(factory)
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

    public override Task Save(ITensorStore store)
    {
        return Task.WhenAll(modules.Select(module => module.Save(store)));
    }

    public override Task Load(ITensorStore store)
    {
        return Task.WhenAll(modules.Select(module => module.Load(store)));
    }

    public override void AddModule(NNModule<T> module)
    {
        modules.Add(module);
    }
}