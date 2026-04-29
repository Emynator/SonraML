using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class Sigmoid<T> : NNModule<T> where T : struct
{
    public override void Dispose()
    {
    }
    
    public override void Forward(Tensor<T> x)
    {
        x.Sigmoid();
    }
}