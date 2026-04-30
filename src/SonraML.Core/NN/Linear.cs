using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class Linear<T> : NNModule<T> where T : struct
{
    private readonly IGlobalTensorFactory gtf;
    private readonly IScopedTensorFactory tf;
    private readonly Tensor<T> weights;
    private readonly Tensor<T> biases;

    public Linear(IGlobalTensorFactory gtf, IScopedTensorFactory tf, int inputFeatures, int outputFeatures)
    {
        this.gtf = gtf;
        this.tf = tf;
        
        var weightShape = new TensorShape([inputFeatures, outputFeatures]);
        var weightArray = new T[weightShape.Size];
        weights = gtf.Create(weightArray.AsMemory(), weightShape);
        
        var biasesShape = new TensorShape([outputFeatures]);
        var biasesArray = new T[biasesShape.Size];
        biases = gtf.Create(biasesArray.AsMemory(), biasesShape);
    }

    public override Tensor<T> Forward(Tensor<T> x)
    {
        return x.Fma(weights, biases);
    }
}