using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public sealed class ReLU<T> : NNModule<T> where T : struct
{
    private readonly Tensor<T> zero;
    private Tensor<bool>? mask;

    public ReLU(IGlobalTensorFactory tf)
    {
        zero = tf.ScalarZero<T>();
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        mask = input.Greater(zero);
        
        return input.Maximum(zero);
    }

    public override Tensor<T> Backward(Tensor<T> gradOutput)
    {
        if (mask is null)
        {
            throw new InvalidOperationException("Cannot call Backward before Forward.");
        }

        var zeros = gradOutput.ZerosLike();

        return mask.Where(gradOutput, zeros);
    }
}