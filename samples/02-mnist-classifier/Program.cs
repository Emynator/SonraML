using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using SonraML.Backend.MLX;
using SonraML.Core;
using SonraML.Core.Config;
using SonraML.Core.Enums;
using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.NN;
using SonraML.Core.Services;
using SonraML.Core.Types;

var builder = Host.CreateApplicationBuilder(args);
builder.ConfigureSonraML(MlxBackendConfiguration.UseMlxBackend);
builder.Services.AddRunner<MnistRunner, MnistRunnerContext>();
builder.Services.Configure<SonraRunnerConfigurations>(options =>
{
    options.Configurations.Add(new SonraRunnerConfiguration
    {
        RunnerName = nameof(MnistRunner),
        Epochs = MnistSampleSettings.MaxEpochs,
    });
});

using var host = builder.Build();
host.InitSonraML(BackendDeviceType.Gpu);
await host.RunAsync();

public sealed class MnistRunner : SonraRunner
{
    private readonly IHostApplicationLifetime lifetime;
    private readonly ILogger<MnistRunner> logger;
    private readonly SgdOptimizer<float> optimizer;
    private readonly IScopedTensorFactory tf;

    public MnistRunner(
        IHostApplicationLifetime lifetime,
        ILogger<MnistRunner> logger,
        IScopedTensorFactory tf)
    {
        this.lifetime = lifetime;
        this.logger = logger;
        this.tf = tf;
        optimizer = new SgdOptimizer<float>(tf, 0.0001f);
    }

    public override async Task Run(CancellationToken ct)
    {
        var context = (MnistRunnerContext)Context;
        var batch = await context.DataLoader.GetData(MnistSampleSettings.BatchSize);
        context.DataLoader.Prefetch(MnistSampleSettings.BatchSize);

        var inputs = batch
            .Select(item => tf.FromArray(item.Input, new TensorShape(new[] { item.Input.Length })))
            .ToArray();
        var expected = batch
            .Select(item => tf.FromArray(item.ExpectedOutput, new TensorShape(new[] { item.ExpectedOutput.Length })))
            .ToArray();

        var input = tf.Stack(inputs, 0, "mnist_input");
        var target = tf.Stack(expected, 0, "mnist_target");
        var prediction = context.Module.Forward(input);
        var loss = Losses.MeanSquaredError(tf, prediction, target, out var gradient);

        gradient.EnsureCompute();
        context.Module.Backward(gradient).EnsureCompute();
        optimizer.Step(context.Module.Parameters);

        context.Epoch++;
        logger.LogInformation("Epoch {Epoch}: loss {Loss}", context.Epoch, loss);

        if (context.Epoch >= MnistSampleSettings.MaxEpochs)
        {
            lifetime.StopApplication();
        }
    }
}

public static class MnistSampleSettings
{
    public const int MaxEpochs = 3;
    public const int BatchSize = 100;
}

public sealed class MnistRunnerContext : ISonraRunnerContext
{
    public MnistRunnerContext(ModuleFactory factory)
    {
        var dataset = new MnistDataset(
            ResolveAsset("train-images.idx3-ubyte"),
            ResolveAsset("train-labels.idx1-ubyte"));

        DataLoader = new MnistDataLoader(dataset);
        Module = new MnistModule(factory, dataset.ImageSize);
    }

    public MnistDataLoader DataLoader { get; }

    public MnistModule Module { get; }

    public int Epoch { get; set; }

    private static string ResolveAsset(string fileName)
    {
        var paths = new[]
        {
            Path.Combine(Environment.CurrentDirectory, "Assets", fileName),
            Path.Combine(AppContext.BaseDirectory, "Assets", fileName),
        };

        return paths.FirstOrDefault(File.Exists)
            ?? throw new FileNotFoundException($"Copy MNIST asset '{fileName}' into an Assets folder next to this sample.");
    }
}

public sealed class MnistModule : NNModule<float>
{
    private readonly Sequential<float> sequential;

    public MnistModule(ModuleFactory factory, int imageSize)
    {
        sequential = factory.CreateSequential<float>();
        sequential
            .AddLinear(imageSize, 16, "hidden1")
            .AddReLU()
            .AddLinear(16, 16, "hidden2")
            .AddReLU()
            .AddLinear(16, 10, "output");
    }

    public override IEnumerable<Parameter<float>> Parameters => sequential.Parameters;

    public override Tensor<float> Forward(Tensor<float> input) => sequential.Forward(input);

    public override Tensor<float> Backward(Tensor<float> gradOutput) => sequential.Backward(gradOutput);
}

public sealed class MnistDataLoader : DataLoader<MnistTrainingData>
{
    private readonly MnistDataset dataset;

    public MnistDataLoader(MnistDataset dataset)
    {
        this.dataset = dataset;
    }

    protected override Task<IEnumerable<MnistTrainingData>> Load(int amount)
    {
        var result = new List<MnistTrainingData>(amount);
        for (var i = 0; i < amount; i++)
        {
            var image = dataset.GetNext();
            var input = image.Pixels.Select(pixel => pixel / 255.0f).ToArray();
            var expected = new float[10];
            expected[image.Label] = 1.0f;
            result.Add(new MnistTrainingData(input, expected));
        }

        return Task.FromResult<IEnumerable<MnistTrainingData>>(result);
    }
}

public sealed class MnistDataset
{
    private readonly byte[] images;
    private readonly byte[] labels;
    private int index;

    public MnistDataset(string imagesPath, string labelsPath)
    {
        using var imageReader = new BinaryReader(File.OpenRead(imagesPath));
        using var labelReader = new BinaryReader(File.OpenRead(labelsPath));

        var imageMagic = ReadBigEndianInt32(imageReader);
        var imageCount = ReadBigEndianInt32(imageReader);
        var rows = ReadBigEndianInt32(imageReader);
        var columns = ReadBigEndianInt32(imageReader);
        var labelMagic = ReadBigEndianInt32(labelReader);
        var labelCount = ReadBigEndianInt32(labelReader);

        if (imageMagic != 2051 || labelMagic != 2049 || imageCount != labelCount)
        {
            throw new InvalidDataException("The MNIST image and label files do not contain matching IDX data.");
        }

        ImageSize = rows * columns;
        Count = imageCount;
        images = imageReader.ReadBytes(Count * ImageSize);
        labels = labelReader.ReadBytes(Count);
    }

    public int Count { get; }

    public int ImageSize { get; }

    public MnistImage GetNext()
    {
        var offset = index * ImageSize;
        var pixels = images[offset..(offset + ImageSize)];
        var label = labels[index];
        index = (index + 1) % Count;

        return new MnistImage(pixels, label);
    }

    private static int ReadBigEndianInt32(BinaryReader reader)
    {
        var bytes = reader.ReadBytes(sizeof(int));
        if (BitConverter.IsLittleEndian)
        {
            Array.Reverse(bytes);
        }

        return BitConverter.ToInt32(bytes);
    }
}

public sealed record MnistImage(byte[] Pixels, byte Label);

public sealed record MnistTrainingData(float[] Input, float[] ExpectedOutput);

