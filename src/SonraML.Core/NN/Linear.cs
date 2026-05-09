using Microsoft.Extensions.Logging;
using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public sealed class Linear<T> : NNModule<T> where T : struct
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

        var w = gtf.FromArray(weightArray, weightShape, $"{name}_weights").ConvertTo<T>();
        weights = new(w, gtf.Zero<T>(weightShape), w.Name);

        var biasesShape = new TensorShape([outputFeatures]);
        var biasesArray = new float[biasesShape.Size];
        for (var i = 0; i < biasesShape.Size; i++)
        {
            biasesArray[i] = Random.Shared.NextSingle() * 10.0f - 5.0f;
        }

        var b = gtf.FromArray(biasesArray, biasesShape, $"{name}_biases").ConvertTo<T>();
        biases = new(b, gtf.Zero<T>(biasesShape), b.Name);
    }

    public override IEnumerable<Parameter<T>> Parameters
    {
        get
        {
            yield return weights;
            yield return biases;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        cachedInput = input;

        return input.Fma(weights.Value, biases.Value);
    }

    public override Tensor<T> Backward(Tensor<T> gradOutput)
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

    public override async Task Save(ITensorStore store)
    {
        if (!canSave)
        {
            return;
        }

        await store.AddTensors([weights.Value, biases.Value]);
    }

    public override async Task Load(ITensorStore store)
    {
        if (!canSave)
        {
            return;
        }

        var w = await store.LoadTensor(weights.Name);
        if (w is not null)
        {
            weights.Value.CopyFrom(w.AsTensor<T>());
        }
        
        var b = await store.LoadTensor(biases.Name);
        if (b is not null)
        {
            biases.Value.CopyFrom(b.AsTensor<T>());
        }
    }
}