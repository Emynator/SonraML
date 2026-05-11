# SonraML samples

This folder contains runnable samples that show how to use SonraML as a .NET-first machine learning framework. The samples are ordered as a learning path: start with tensors, then move to training, persistence, custom modules, and .NET-native data pipelines.

## Prerequisites

- .NET 10 SDK.
- MLX-C native libraries copied to `src/SonraML.Backend.MLX/NativeLibs`.
- Apple Silicon/macOS is the primary tested target for the MLX backend.
- MNIST assets are required only for `02-mnist-classifier`.

## Samples

| Folder | Topic | What it demonstrates |
| --- | --- | --- |
| `01-tensor-basics` | Tensor fundamentals | Creating tensors, using shapes, arithmetic, reductions, LINQ enumeration, and explicit `EnsureCompute()` calls. |
| `02-mnist-classifier` | End-to-end training | Generic Host setup, MLX backend initialization, `DataLoader<T>`, `Sequential<float>`, MSE loss, manual backward pass, and SGD. |
| `03-save-load-safetensors` | Model persistence | Saving named module parameters to `.safetensors`, listing stored tensors, and loading them into a new model instance. |
| `04-custom-module-xor` | Custom modules | Creating an `NNModule<float>` wrapper around reusable SonraML layers for a small XOR classifier. |
| `05-dotnet-data-pipeline` | .NET data loading | Using records, `IEnumerable<T>`, LINQ transforms, batching, and background prefetching with `DataLoader<T>`. |

## Running a sample

Run samples from the repository root:

```powershell
dotnet run --project .\samples\01-tensor-basics\TensorBasics.Sample.csproj
```

Replace the project path with the sample you want to run.

## Validated external references

- [MLX](https://github.com/ml-explore/mlx) is an array framework for machine learning on Apple silicon. Its documentation describes lazy computation and CPU/GPU support.
- [MLX-C](https://github.com/ml-explore/mlx-c) is the C API for MLX and is the native dependency used by the MLX backend.
- [Safetensors](https://github.com/huggingface/safetensors) is a simple tensor storage format designed to be safer than pickle and fast to load.
- [MNIST](http://yann.lecun.com/exdb/mnist/) is the original handwritten digit dataset. The classifier sample expects the standard training image and label IDX files after extraction.

