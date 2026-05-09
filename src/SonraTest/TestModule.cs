using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.NN;
using SonraML.Core.Services;
using SonraML.Core.Types;

namespace SonraTest;

public class TestModule : NNModule<float>
{
    private readonly Sequential<float> sequential;
    
    public TestModule(ModuleFactory factory, int imageSize)
    {
        sequential = factory.CreateSequential<float>();
        sequential
            .AddLinear(imageSize, 16)
            .AddReLU()
            .AddLinear(16, 16)
            .AddReLU()
            .AddLinear(16, 10);
    }
    
    public override IEnumerable<Parameter<float>> Parameters => sequential.Parameters;

    public override Tensor<float> Forward(Tensor<float> input)
    {
        return sequential.Forward(input);
    }

    public override Tensor<float> Backward(Tensor<float> gradOutput)
    {
        return sequential.Backward(gradOutput);
    }

    public override Task Save(ITensorStore store)
    {
        return sequential.Save(store);
    }

    public override Task Load(ITensorStore store)
    {
        return sequential.Load(store);
    }
}