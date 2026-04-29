using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class Sigmoid<T> : NNModule<T> where T : struct
{
    public override Tensor<T> Forward(Tensor<T> x)
    {
        // TODO
        throw new NotImplementedException();
    }
}