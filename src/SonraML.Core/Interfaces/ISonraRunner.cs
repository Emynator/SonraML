namespace SonraML.Core.Interfaces;

public interface ISonraRunner
{
    public ISonraRunnerContext Context { get; set; }
    
    public Task Run(CancellationToken ct);
}