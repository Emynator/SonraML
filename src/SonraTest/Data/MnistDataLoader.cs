namespace SonraTest.Data;

public class MnistDataLoader
{
    private readonly MnistDataset dataset;

    public MnistDataLoader(MnistDataset dataset)
    {
        this.dataset = dataset;
    }

    public Task<MnistTrainingData> GetData(int batchSize)
    {
        var tcs = new TaskCompletionSource<MnistTrainingData>();
        Task.Run(() => GetData(tcs, batchSize));

        return tcs.Task;
    }

    private Task GetData(TaskCompletionSource<MnistTrainingData> tcs, int batchSize)
    {
        var inputs = new List<float[]>(batchSize);
        var outputs = new List<float[]>(batchSize);

        for (var i = 0; i < batchSize; i++)
        {
            var data = dataset.GetNext();
            if (data is null)
            {
                break;
            }

            inputs.Add(data.Image.Select(im => im / 255.0f).ToArray());
            var output = new float[10];
            output[data.Label] = 1.0f;
            outputs.Add(output);
        }

        tcs.SetResult(new(inputs, outputs));

        return Task.CompletedTask;
    }
}