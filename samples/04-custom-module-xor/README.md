# 04 - Custom module XOR

This sample shows how to wrap SonraML layers in a custom `NNModule<float>`.

## Concepts covered

- Creating a reusable module type by inheriting from `NNModule<float>`.
- Delegating `Forward`, `Backward`, and `Parameters` to an internal `Sequential<float>`.
- Training against the XOR truth table.
- Using one-hot targets and `MeanSquaredError`.
- Updating parameters with `SgdOptimizer<float>`.

## Run

```powershell
dotnet run --project .\samples\04-custom-module-xor\CustomModuleXor.Sample.csproj
```

The sample performs a short training loop and prints the loss after each epoch.

