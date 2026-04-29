using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class Linear<T> : NNModule<T> where T : struct
{
    private readonly Tensor<T> weights;
    private readonly Tensor<T> biases;

    public Linear(int dimensions, int features)
    {
        var weightShape = new TensorShape([dimensions, features]);
        var weightArray = new T[weightShape.Size];
        weights = Tf.Create(weightArray.AsMemory(), weightShape);
        
        var biasesShape = new TensorShape([dimensions]);
        var biasesArray = new T[biasesShape.Size];
        biases = Tf.Create(biasesArray.AsMemory(), biasesShape);
    }

    public override void Dispose()
    {
        weights.Dispose();
        biases.Dispose();
    }

    public override void Forward(Tensor<T> x)
    {
        x.Fma(weights, biases);
    }
}