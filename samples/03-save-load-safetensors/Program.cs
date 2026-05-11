using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Backend.MLX;
using SonraML.Core;
using SonraML.Core.Enums;
using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.IO;
using SonraML.Core.NN;
using SonraML.Core.Services;

var builder = Host.CreateApplicationBuilder(args);
builder.ConfigureSonraML(MlxBackendConfiguration.UseMlxBackend);

using var host = builder.Build();
host.InitSonraML(BackendDeviceType.Gpu);

using var scope = host.Services.CreateScope();
var tf = scope.ServiceProvider.GetRequiredService<IScopedTensorFactory>();
var moduleFactory = scope.ServiceProvider.GetRequiredService<ModuleFactory>();

var checkpointDirectory = Path.Combine(Environment.CurrentDirectory, "samples", "03-save-load-safetensors", "checkpoints");
Directory.CreateDirectory(checkpointDirectory);

var checkpointPath = Path.Combine(checkpointDirectory, "classifier.safetensors");
if (File.Exists(checkpointPath))
{
    File.Delete(checkpointPath);
}

var model = CreateModel(moduleFactory);
await using (var store = new SafetensorsStore(tf, checkpointPath))
{
    await model.Save(store);
    await store.Persist();
    var tensorNames = await store.ListTensors();
    Console.WriteLine($"Saved {tensorNames.Count} tensors: {string.Join(", ", tensorNames)}");
}

var reloaded = CreateModel(moduleFactory);
await using (var store = new SafetensorsStore(tf, checkpointPath))
{
    await reloaded.Load(store);
    var tensorNames = await store.ListTensors();
    Console.WriteLine($"Loaded checkpoint from {checkpointPath}");
    Console.WriteLine($"Checkpoint contains: {string.Join(", ", tensorNames)}");
}

static Sequential<float> CreateModel(ModuleFactory factory)
{
    var model = factory.CreateSequential<float>();
    model
        .AddLinear(4, 8, "features")
        .AddReLU()
        .AddLinear(8, 3, "classifier");

    return model;
}

