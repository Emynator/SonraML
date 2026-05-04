# SonraML
A Backend-Agnostic Machine Learning Framework for .NET

---

SonraML is a .NET machine learning framework with backend-agnostic tensors, scoped tensor lifetimes, neural-network modules, optimizers, safetensors support, and an MLX backend for Apple Silicon.

Define your ML modules with fluent syntax:
```C#
var sequential = new(serviceProvider);
sequential
    .AddLinear(imageSize, 16)
    .AddReLU()
    .AddLinear(16, 16)
    .AddReLU()
    .AddLinear(16, 10);
```

Integrates with .NET's `IHost` model:
```C#
var builder = Host.CreateApplicationBuilder(args);
builder.ConfigureSonraML(MlxBackendConfiguration.UseMlxBackend);
builder.Services.AddRunner<TestRunner, TestRunnerContext>();

var host = builder.Build();
host.InitSonraML(BackendDeviceType.Gpu);
host.Run();
```

## Why SonraML?
Most modern ML tooling assumes Python first. SonraML explores what a native .NET ML framework can look like:
strong typing, DI integration, backend abstraction, scoped lifetimes, and C#-friendly model composition.

## What makes SonraML different?
SonraML is built around the .NET application model. Tensors are created through dependency injection, scoped lifetimes clean up native resources, runners execute through `IHost`, and data can come from ordinary .NET sources such as files, streams, LINQ, or EF Core.

## Project status
- SonraML is experimental and early.
- The MLX backend is currently the only backend.
- The API may change.
- Training currently uses manually implemented backward passes for supported modules.

## Features
- **Backend agnostic tensors:** Generic tensor class abstracts backend implementations away.
- **Statically typed and type-safe tensors:** because a `Tensor<float>` should not be confused with a `Tensor<bool>`.
- **Modular architecture:** Extend features with custom `INNModule<T>` implementations from other sources.
- **DI scoped tensor lifetimes:** Inject an `IScopedTensorFactory` or `IGlobalTensorFactory` via dependency injection, create tensors and don't worry about native memory management.
**Lazy evaluation for compute graphs:** Computation is deferred until data is accessed, copied, printed, or explicitly evaluated with `EnsureCompute()`.
- **LINQ up your data:** If .NET can IEnumerable it, SonraML can train on it. EF Core data providers anyone?
**`Tensor<T>` implements `IEnumerable<T>`:** Use LINQ for inspection and CPU-side processing. Enumeration reads from backend memory without forcing an intermediate managed array unless you create one.
- **Let SonraML manage your training runs:** Implement and configure a `SonraRunner` and SonraML will execute training runs for you. Each epoch is executed in its own DI scope that is disposed after execution is finished.
- **Let SonraML load your training data:** `DataLoader<T>` handles batching and background prefetching. You just provide the data source.

## Building
Building and using SonraML requires MLX-C libraries. Check out [the GitHub repo](https://github.com/ml-explore/mlx-c) for build instructions. Copy the resulting .dylibs to `src/SonraML.Backend.MLX/NativeLibs` and it should work. `SonraTest` requires the MNIST dataset in `src/SonraTest/Assets`.

SonraML was tested on macOS with MLX-C. It *might* work on Linux and Windows since MLX *does* have a CUDA backend, but I wasn't able to verify that. Contributions on that front are more than welcome!

## Contributing
SonraML is still early and experimental, so just reach out!