using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.NN;
using SonraML.Core.Types;

namespace SonraTest;

public class TestRunner : SonraRunner
{
    public static Tensor<float>? Tensor;

    private readonly ILogger<TestRunner> logger;
    private readonly IScopedTensorFactory tf;
    private readonly SgdOptimizer<float> optimizer;

    public TestRunner(ILogger<TestRunner> logger, IScopedTensorFactory tf)
    {
        this.logger = logger;
        this.tf = tf;
        optimizer = new(tf, 0.0001f);
    }

    public override async Task Run(CancellationToken ct)
    {
        var context = Context as TestRunnerContext;
        if (context is null)
        {
            throw new InvalidOperationException("Context is null!");
        }

        var data = await context.DataLoader.GetData(100);
        var inputs = data.Inputs
            .Select(i => tf.FromArray(i, new([i.Length])))
            .ToArray();
        var input = tf.Stack(inputs, 0, null);
        
        var outputs = data.ExpectedOutputs
            .Select(o => tf.FromArray(o, new([o.Length])))
            .ToArray();
        var output = tf.Stack(outputs, 0, null);
        
        var result = context.Module.Forward(input);
        
        var error = Losses.MeanSquaredError
        (
            tf,
            result,
            output,
            out var gradient
        );
        gradient.EnsureCompute();
        var errorString = error.ToString();
        
        var res = context.Module.Backward(gradient);
        res.EnsureCompute();

        optimizer.Step(context.Module.Parameters);
        logger.LogInformation("Error: {Error}", errorString);
    }
}