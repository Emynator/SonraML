# 02 - MNIST classifier

This sample trains a small fully connected network on MNIST handwritten digits.

## Concepts covered

- Integrating SonraML with `Host.CreateApplicationBuilder`.
- Initializing the MLX backend with `host.InitSonraML(BackendDeviceType.Gpu)`.
- Registering a `SonraRunner` and persistent `ISonraRunnerContext`.
- Loading binary IDX data into a custom `DataLoader<T>`.
- Building a `Sequential<float>` model with `Linear` and `ReLU` layers.
- Computing mean squared error, calling `Backward`, and updating parameters with `SgdOptimizer<float>`.

## Data

Download the standard MNIST training files from the original MNIST site and extract them into:

```text
samples/02-mnist-classifier/Assets/train-images.idx3-ubyte
samples/02-mnist-classifier/Assets/train-labels.idx1-ubyte
```

The source files are named `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz`.

## Run

```powershell
dotnet run --project .\samples\02-mnist-classifier\MnistClassifier.Sample.csproj
```

The sample runs three epochs and logs the loss for each epoch.

