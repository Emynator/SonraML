using SonraML.Core.Interfaces;

namespace SonraML.Core.Types;

public abstract class SonraRunner : IDisposable
{
    public ISonraRunnerContext Context { get; set; } = null!;

    public virtual void Dispose()
    {
    }

    public virtual Task BeforeFirstRun(CancellationToken ct)
    {
        return Task.CompletedTask;
    }

    public abstract Task Run(CancellationToken ct);
}