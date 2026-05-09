using SonraML.Core.Interfaces;
using SonraML.Core.Services;
using SonraTest.Data;

namespace SonraTest;

public class TestRunnerContext : ISonraRunnerContext
{
    public TestRunnerContext(ModuleFactory factory)
    {
        Console.WriteLine(Directory.GetCurrentDirectory());
        var dataset = new MnistDataset
        (
            "Assets/train-images.idx3-ubyte",
            "Assets/train-labels.idx1-ubyte"
        );
        
        DataLoader = new(dataset);
        Module = new(factory, dataset.ImageSize);
    }
    
    public MnistDataLoader DataLoader { get; }
    
    public TestModule Module { get; }
}