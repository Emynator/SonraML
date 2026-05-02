using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.NN;
using SonraML.Core.Types;

namespace SonraTest;

public class TestModule : INNModule<float>
{
    private readonly Sequential<float> sequential;
    
    public TestModule(IServiceProvider serviceProvider, int imageSize)
    {
        sequential = new(serviceProvider);
        sequential
            .AddLinear(imageSize, 16)
            .AddReLU()
            .AddLinear(16, 16)
            .AddReLU()
            .AddLinear(16, 10);
    }
    
    public IEnumerable<Parameter<float>> Parameters => sequential.Parameters;

    public Tensor<float> Forward(Tensor<float> input)
    {
        return sequential.Forward(input);
    }

    public Tensor<float> Backward(Tensor<float> gradOutput)
    {
        return sequential.Backward(gradOutput);
    }
}