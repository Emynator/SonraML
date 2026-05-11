# 01 - Tensor basics

This sample introduces SonraML tensors without defining a neural network.

## Concepts covered

- Configuring SonraML with the MLX backend.
- Resolving an `IScopedTensorFactory` from dependency injection.
- Creating tensors from .NET arrays with explicit `TensorShape` values.
- Running tensor arithmetic, broadcasting, `Square()`, and `Mean(axis: 0)`.
- Calling `EnsureCompute()` to force lazy MLX work to materialize.
- Enumerating `Tensor<T>` with LINQ for CPU-side inspection.

## Run

```powershell
dotnet run --project .\samples\01-tensor-basics\TensorBasics.Sample.csproj
```

The sample prints each tensor shape and values, then uses LINQ to sum the shifted tensor values.

