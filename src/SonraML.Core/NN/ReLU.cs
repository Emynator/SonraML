using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class ReLU<T> : NNModule<T> where T : struct
{
    private readonly IScopedTensorFactory tf;
    private readonly Tensor<T> zero;

    public ReLU(IScopedTensorFactory tf)
    {
        this.tf = tf;
        zero = tf.ScalarZero<T>();
    }

    public override Tensor<T> Forward(Tensor<T> x)
    {
        return x.Maximum(zero);
    }
}