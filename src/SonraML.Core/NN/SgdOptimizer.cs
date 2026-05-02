using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public class SgdOptimizer<T> : IOptimizer<T> where T : struct
{
    private readonly Tensor<T> learningRate;
    
    public SgdOptimizer(ITensorFactory tf, T learningRate)
    {
        this.learningRate = tf.Create(learningRate);
    }
    
    public void Step(IEnumerable<Parameter<T>> parameters)
    {
        Tensor<T>? adjustment = null;
        foreach (var parameter in parameters)
        {
            if (!parameter.HasGradient)
            {
                continue;
            }

            adjustment = parameter.Gradient * learningRate;
            parameter.SetValue(parameter.Value - adjustment);
            parameter.ClearGradient();
        }
        
        adjustment?.EnsureCompute();
    }
}