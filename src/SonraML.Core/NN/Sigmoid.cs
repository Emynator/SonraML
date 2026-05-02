using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class Sigmoid<T> : INNModule<T> where T : struct
{
    public IEnumerable<Parameter<T>> Parameters
    {
        get
        {
            yield break;
        }
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        return input.Sigmoid();
    }

    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        throw new NotImplementedException();
    }
}