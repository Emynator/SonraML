namespace SonraML.Core.Interfaces;

public interface ISonraRunner
{
    public Task Run(CancellationToken ct);
}