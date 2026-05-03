using SonraML.Core.Types;

namespace SonraTest.Data;

public class MnistDataLoader : DataLoader<MnistTrainingData>
{
    private readonly MnistDataset dataset;

    public MnistDataLoader(MnistDataset dataset)
    {
        this.dataset = dataset;
    }

    protected override Task<IEnumerable<MnistTrainingData>> Load(int amount)
    {
        var result = new List<MnistTrainingData>();

        for (var i = 0; i < amount; i++)
        {
            var data = dataset.GetNext();
            if (data is null)
            {
                break;
            }

            var input = data.Image.Select(im => im / 255.0f).ToArray();
            var output = new float[10];
            output[data.Label] = 1.0f;
            
            result.Add(new(input, output));
        }
        
        return Task.FromResult<IEnumerable<MnistTrainingData>>(result);
    }
}