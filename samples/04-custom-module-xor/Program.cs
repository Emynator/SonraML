using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Backend.MLX;
using SonraML.Core;
using SonraML.Core.Enums;
using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.NN;
using SonraML.Core.Services;
using SonraML.Core.Types;

var builder = Host.CreateApplicationBuilder(args);
builder.ConfigureSonraML(MlxBackendConfiguration.UseMlxBackend);

using var host = builder.Build();
host.InitSonraML(BackendDeviceType.Gpu);

using var scope = host.Services.CreateScope();
var tf = scope.ServiceProvider.GetRequiredService<IScopedTensorFactory>();
var moduleFactory = scope.ServiceProvider.GetRequiredService<ModuleFactory>();

var classifier = new XorClassifier(moduleFactory);
var optimizer = new SgdOptimizer<float>(tf, 0.01f);
var input = tf.FromArray(
    new[] { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f },
    new TensorShape(new[] { 4, 2 }),
    "xor_input");
var expected = tf.FromArray(
    new[] { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f },
    new TensorShape(new[] { 4, 2 }),
    "xor_expected");

for (var epoch = 1; epoch <= 10; epoch++)
{
    var prediction = classifier.Forward(input);
    var loss = Losses.MeanSquaredError(tf, prediction, expected, out var gradient);
    gradient.EnsureCompute();

    classifier.Backward(gradient).EnsureCompute();
    optimizer.Step(classifier.Parameters);

    Console.WriteLine($"Epoch {epoch}: loss {loss}");
}

public sealed class XorClassifier : NNModule<float>
{
    private readonly Sequential<float> layers;

    public XorClassifier(ModuleFactory factory)
    {
        layers = factory.CreateSequential<float>();
        layers
            .AddLinear(2, 4, "xor_hidden")
            .AddReLU()
            .AddLinear(4, 2, "xor_output");
    }

    public override IEnumerable<Parameter<float>> Parameters => layers.Parameters;

    public override Tensor<float> Forward(Tensor<float> input) => layers.Forward(input);

    public override Tensor<float> Backward(Tensor<float> gradOutput) => layers.Backward(gradOutput);
}

