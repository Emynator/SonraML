# 03 - Save and load Safetensors

This sample demonstrates model checkpointing with SonraML's `ITensorStore` abstraction and `SafetensorsStore`.

## Concepts covered

- Creating named layers so parameters can be persisted.
- Saving module parameters to a `.safetensors` file.
- Listing tensors stored in a checkpoint.
- Loading parameters into a new model instance with the same parameter names.

## Run

```powershell
dotnet run --project .\samples\03-save-load-safetensors\SaveLoadSafetensors.Sample.csproj
```

The sample writes `samples/03-save-load-safetensors/checkpoints/classifier.safetensors` and then loads it into a second model instance.

