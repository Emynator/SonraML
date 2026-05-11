# 05 - .NET data pipeline

This sample focuses on data loading rather than model training.

## Concepts covered

- Modeling source data with C# records.
- Using `IEnumerable<T>` and LINQ as the input data source.
- Creating a custom `DataLoader<T>` that transforms domain records into training rows.
- Scheduling background prefetch with `Prefetch`.
- Converting batched records into feature and label tensors.

## Run

```powershell
dotnet run --project .\samples\05-dotnet-data-pipeline\DotNetDataPipeline.Sample.csproj
```

The sample prints the resulting feature and label tensor shapes for the prefetched batch.

