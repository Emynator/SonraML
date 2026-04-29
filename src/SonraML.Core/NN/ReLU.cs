using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class ReLU<T> : NNModule<T> where T : struct
{
    private readonly Tensor<T> zero = SonraMLConfiguration.Backend.TensorFactory.ScalarZero<T>();
    
    public override void Dispose()
    {
        zero.Dispose();
    }

    public override void Forward(Tensor<T> x)
    {
        x.Maximum(zero);
    }
}