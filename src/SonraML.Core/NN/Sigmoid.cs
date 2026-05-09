using SonraML.Core.Types;

namespace SonraML.Core.NN;

public sealed class Sigmoid<T> : NNModule<T> where T : struct
{
    public override Tensor<T> Forward(Tensor<T> input)
    {
        return input.Sigmoid();
    }

    public override Tensor<T> Backward(Tensor<T> gradOutput)
    {
        throw new NotImplementedException();
    }
}