using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraTest;

public class TestRunner : ISonraRunner
{
    public static Tensor<float>? Tensor;
    
    private readonly ILogger<TestRunner> logger;
    private readonly IGlobalTensorFactory gtf;
    private readonly IScopedTensorFactory tf;

    public TestRunner(ILogger<TestRunner> logger, IGlobalTensorFactory gtf, IScopedTensorFactory tf)
    {
        this.logger = logger;
        this.gtf = gtf;
        this.tf = tf;
    }
    
    public async Task Run(CancellationToken ct)
    {
        if (Tensor is null)
        {
            Tensor = gtf.Zero<float>(new([5, 5]));
        }
        
        var toAdd = tf.One<float>(new([5, 5]));
        Tensor += toAdd;
        
        logger.LogInformation(Tensor.ToString());
        
        await Task.Delay(1000, ct);
    }
}