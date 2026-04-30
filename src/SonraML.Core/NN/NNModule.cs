using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public abstract class NNModule<T> where T : struct
{
    public abstract Tensor<T> Forward(Tensor<T> x);
}