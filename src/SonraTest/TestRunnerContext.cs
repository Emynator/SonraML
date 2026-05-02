using SonraML.Core.Interfaces;
using SonraTest.Data;

namespace SonraTest;

public class TestRunnerContext : ISonraRunnerContext
{
    public TestRunnerContext(IServiceProvider serviceProvider)
    {
        var dataset = new MnistDataset
        (
            "/Users/emily/Development/SonraML/TestData/train-images.idx3-ubyte",
            "/Users/emily/Development/SonraML/TestData/train-labels.idx1-ubyte"
        );
        
        DataLoader = new(dataset);
        Module = new(serviceProvider, dataset.ImageSize);
    }
    
    public MnistDataLoader DataLoader { get; }
    
    public TestModule Module { get; }
}