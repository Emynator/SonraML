using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Backend.MLX;
using SonraML.Core;
using SonraML.Core.Enums;
using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

var source = Enumerable.Range(0, 32)
    .Select(index => new SensorReading(
        TemperatureCelsius: 18.0f + index * 0.25f,
        HumidityPercent: 40.0f + index % 10,
        IsComfortable: index % 3 != 0))
    .ToArray();

var loader = new SensorReadingLoader(source);
loader.Prefetch(8);

var builder = Host.CreateApplicationBuilder(args);
builder.ConfigureSonraML(MlxBackendConfiguration.UseMlxBackend);

using var host = builder.Build();
host.InitSonraML(BackendDeviceType.Gpu);

using var scope = host.Services.CreateScope();
var tf = scope.ServiceProvider.GetRequiredService<IScopedTensorFactory>();

var batch = await loader.GetData(8);
var featureTensors = batch
    .Select(item => tf.FromArray(item.Features, new TensorShape(new[] { item.Features.Length })))
    .ToArray();
var labelTensors = batch
    .Select(item => tf.FromArray(new[] { item.Label }, new TensorShape(new[] { 1 })))
    .ToArray();

var features = tf.Stack(featureTensors, 0, "sensor_features");
var labels = tf.Stack(labelTensors, 0, "sensor_labels");

features.EnsureCompute();
labels.EnsureCompute();

Console.WriteLine($"Loaded {batch.Count} records.");
Console.WriteLine($"Feature tensor shape: [{string.Join(", ", features.Shape.Shape)}]");
Console.WriteLine($"Label tensor shape: [{string.Join(", ", labels.Shape.Shape)}]");

public sealed class SensorReadingLoader : DataLoader<TrainingRow>
{
    private readonly IEnumerator<SensorReading> enumerator;

    public SensorReadingLoader(IEnumerable<SensorReading> readings)
    {
        enumerator = readings.GetEnumerator();
    }

    protected override Task<IEnumerable<TrainingRow>> Load(int amount)
    {
        var rows = new List<TrainingRow>(amount);
        for (var i = 0; i < amount && enumerator.MoveNext(); i++)
        {
            var reading = enumerator.Current;
            var features = new[]
            {
                reading.TemperatureCelsius / 40.0f,
                reading.HumidityPercent / 100.0f,
            };
            var label = reading.IsComfortable ? 1.0f : 0.0f;
            rows.Add(new TrainingRow(features, label));
        }

        return Task.FromResult<IEnumerable<TrainingRow>>(rows);
    }
}

public sealed record SensorReading(float TemperatureCelsius, float HumidityPercent, bool IsComfortable);

public sealed record TrainingRow(float[] Features, float Label);

