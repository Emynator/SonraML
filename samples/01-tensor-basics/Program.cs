using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Backend.MLX;
using SonraML.Core;
using SonraML.Core.Enums;
using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

var builder = Host.CreateApplicationBuilder(args);
builder.ConfigureSonraML(MlxBackendConfiguration.UseMlxBackend);

using var host = builder.Build();
host.InitSonraML(BackendDeviceType.Gpu);

using var scope = host.Services.CreateScope();
var tf = scope.ServiceProvider.GetRequiredService<IScopedTensorFactory>();

var matrix = tf.FromArray(
    new[] { 1.0f, 2.0f, 3.0f, 4.0f },
    new TensorShape(new[] { 2, 2 }),
    "matrix");
var bias = tf.FromArray(
    new[] { 10.0f, 20.0f },
    new TensorShape(new[] { 2 }),
    "bias");

var shifted = matrix + bias;
var squared = shifted.Square();
var columnMeans = squared.Mean(axis: 0);

columnMeans.EnsureCompute();

PrintTensor("matrix", matrix);
PrintTensor("bias", bias);
PrintTensor("shifted", shifted);
PrintTensor("squared", squared);
PrintTensor("columnMeans", columnMeans);

Console.WriteLine($"LINQ sum of shifted values: {shifted.Sum(v => v):0.###}");

static void PrintTensor<T>(string label, Tensor<T> tensor) where T : struct
{
    var values = tensor.Select(value => Convert.ToString(value, CultureInfo.InvariantCulture));
    Console.WriteLine($"{label} shape=[{string.Join(", ", tensor.Shape.Shape)}] values=[{string.Join(", ", values)}]");
}

