using Microsoft.Extensions.Logging;
using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public sealed class Linear<T> : INNModule<T> where T : struct
{
    private readonly bool canSave = true;
    private readonly Parameter<T> weights;
    private readonly Parameter<T> biases;
    private Tensor<T>? cachedInput;

    public Linear
        (
        ILogger<Linear<T>> logger,
        IGlobalTensorFactory gtf,
        int inputFeatures,
        int outputFeatures,
        string? name
        )
    {
        var weightShape = new TensorShape([inputFeatures, outputFeatures]);
        var weightArray = new float[weightShape.Size];
        for (var i = 0; i < weightShape.Size; i++)
        {
            weightArray[i] = Random.Shared.NextSingle() * 10.0f - 5.0f;
        }

        if (name is null)
        {
            name = "";
            canSave = false;
        }
        
        var w = gtf.FromArray(weightArray, weightShape).ConvertTo<T>();
        weights = new(w, gtf.Zero<T>(weightShape), $"{name}_weights");

        var biasesShape = new TensorShape([outputFeatures]);
        var biasesArray = new float[biasesShape.Size];
        for (var i = 0; i < biasesShape.Size; i++)
        {
            biasesArray[i] = Random.Shared.NextSingle() * 10.0f - 5.0f;
        }
        
        var b = gtf.FromArray(biasesArray, biasesShape).ConvertTo<T>();
        biases = new(b, gtf.Zero<T>(biasesShape), $"{name}_biases");
    }

    public IEnumerable<Parameter<T>> Parameters
    {
        get
        {
            yield return weights;
            yield return biases;
        }
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        cachedInput = input;

        return input.Fma(weights.Value, biases.Value);
    }

    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        if (cachedInput is null)
        {
            throw new InvalidOperationException("Cannot call Backward before Forward.");
        }

        var inputTransposed = cachedInput.Transpose();
        weights.SetGradient(inputTransposed.MatMul(gradOutput));
        biases.SetGradient(gradOutput.Sum(axis: 0));
        var weightsTransposed = weights.Value.Transpose();

        var gradInput = gradOutput.MatMul(weightsTransposed);

        return gradInput;
    }

    public async Task Save(string filePath)
    {
        if (!canSave)
        {
            return;
        }
    }

    public async Task Load(string filePath)
    {
        if (!canSave)
        {
            return;
        }
    }
}