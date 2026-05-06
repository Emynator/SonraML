using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public sealed class ReLU<T> : INNModule<T> where T : struct
{
    private readonly Tensor<T> zero;
    private Tensor<bool>? mask;

    public ReLU(IGlobalTensorFactory tf)
    {
        zero = tf.ScalarZero<T>();
    }

    public IEnumerable<Parameter<T>> Parameters
    {
        get
        {
            yield break;
        }
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        mask = input.Greater(zero);
        
        return input.Maximum(zero);
    }

    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        if (mask is null)
        {
            throw new InvalidOperationException("Cannot call Backward before Forward.");
        }

        var zeros = gradOutput.ZerosLike();

        return mask.Where(gradOutput, zeros);
    }

    public Task Save(ITensorStore store)
    {
        return Task.CompletedTask;
    }

    public Task Load(ITensorStore store)
    {
        return Task.CompletedTask;
    }
}