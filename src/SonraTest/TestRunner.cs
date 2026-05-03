using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.NN;
using SonraML.Core.Types;

namespace SonraTest;

public class TestRunner : ISonraRunner
{
    public static Tensor<float>? Tensor;

    private readonly ILogger<TestRunner> logger;
    private readonly IScopedTensorFactory tf;
    private readonly SgdOptimizer<float> optimizer;

    public TestRunner(ILogger<TestRunner> logger, IScopedTensorFactory tf)
    {
        this.logger = logger;
        this.tf = tf;
        optimizer = new(tf, 0.001f);
    }

    public ISonraRunnerContext Context { get; set; } = null!;

    public async Task Run(CancellationToken ct)
    {
        logger.LogInformation("{time} - Starting new Epoch.", DateTime.Now);
        
        var context = Context as TestRunnerContext;
        if (context is null)
        {
            throw new InvalidOperationException("Context is null!");
        }

        logger.LogInformation("Loading batch...");
        var data = await context.DataLoader.GetData(100);
        logger.LogInformation("{time} - Batch loaded.", DateTime.Now);

        var inputs = data.Inputs
            .Select(i => tf.FromArray(i, new([i.Length])))
            .ToArray();
        var input = tf.Stack(inputs, 0, null);
        
        var outputs = data.ExpectedOutputs
            .Select(o => tf.FromArray(o, new([o.Length])))
            .ToArray();
        var output = tf.Stack(outputs, 0, null);
        
        var result = context.Module.Forward(input);
        logger.LogInformation("{time} - Forward pass done.", DateTime.Now);
        
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
        logger.LogInformation("{time} - Backprop done.", DateTime.Now);

        optimizer.Step(context.Module.Parameters);
        logger.LogInformation("{time} - Epoch done. Error: {Error}", DateTime.Now, errorString);
    }
}