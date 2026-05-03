namespace SonraML.Core.Types;

/// <summary>
/// Generic base class for DataLoaders. Abstracts Prefetching logic away from user code.
/// </summary>
/// <typeparam name="TData">The data type the DataLoader loads.</typeparam>
public abstract class DataLoader<TData> where TData : class
{
    private readonly List<TData> data = [];
    private Task? prefetchTask;
    
    /// <summary>
    /// Schedule a prefetch operation that will be executed on a background thread.
    /// </summary>
    /// <param name="batchSize">Number of items to prefetch.</param>
    public void Prefetch(int batchSize)
    {
        prefetchTask = Task.Run(() => ExecutePrefetch(batchSize));
    }

    /// <summary>
    /// Gets data for a batch. If prefetching is ongoing this will be awaited. If no prefetch was done, data will be gathered.
    /// </summary>
    /// <param name="batchSize">Number of items to get. If batchSize > prefetched data, more data will be loaded..</param>
    /// <returns>ICollection of TData with batchSize items.</returns>
    public async Task<ICollection<TData>> GetData(int batchSize)
    {
        if (prefetchTask is not null)
        {
            await prefetchTask;
            prefetchTask = null;
        }

        var amountMissing = batchSize - data.Count;
        if (amountMissing > 0)
        {
            var missing = await Load(amountMissing);
            data.AddRange(missing);
        }

        var result = data.ToList();
        data.Clear();
        
        return result;
    }

    /// <summary>
    /// Actual data loading function the derived class should implement. 
    /// </summary>
    /// <param name="amount">Number of items to load.</param>
    /// <returns>IEnumerable of TData with amount items.</returns>
    protected abstract Task<IEnumerable<TData>> Load(int amount);

    private async Task ExecutePrefetch(int batchSize)
    {
        var result = await Load(batchSize);
        data.AddRange(result);
    }
}