namespace SonraML.Core.Types;

public abstract class DataLoader<TData> where TData : class
{
    private readonly List<TData> data = [];
    private Task? prefetchTask;
    
    public void Prefetch(int batchSize)
    {
        prefetchTask = Task.Run(() => ExecutePrefetch(batchSize));
    }

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

    protected abstract Task<IEnumerable<TData>> Load(int amount);

    private async Task ExecutePrefetch(int batchSize)
    {
        var result = await Load(batchSize);
        data.AddRange(result);
    }
}