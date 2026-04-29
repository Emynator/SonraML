using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class Linear<T> : NNModule<T> where T : struct
{
    private readonly int featuresIn;
    private readonly int featuresOut;

    public Linear(int featuresIn, int featuresOut)
    {
        this.featuresIn = featuresIn;
        this.featuresOut = featuresOut;
    }
    
    public override Tensor<T> Forward(Tensor<T> x)
    {
        // TODO
        throw new NotImplementedException();
    }
}