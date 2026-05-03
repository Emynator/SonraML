using SonraML.Core.Interfaces;

namespace SonraML.Core.Types;

/// <summary>
/// Abstract base class for all runners. User runner have to inherit from this class.
/// Each epoch uses its own scope and instantiates this class over DI. To store data between epoch runs,
/// use the provided Context.
/// </summary>
public abstract class SonraRunner : IDisposable
{
    /// <summary>
    /// Context that persists between runs. Guaranteed to be set by the worker before user code funcs are executed.
    /// </summary>
    public ISonraRunnerContext Context { get; set; } = null!;

    /// <summary>
    /// Optional: release acquired resources.
    /// </summary>
    public virtual void Dispose()
    {
    }

    /// <summary>
    /// Called once before the first epoch.
    /// </summary>
    /// <param name="ct">CT of the .NET worker service.</param>
    /// <returns>Task for async.</returns>
    public virtual Task BeforeFirstRun(CancellationToken ct)
    {
        return Task.CompletedTask;
    }

    /// <summary>
    /// Called on every epoch.
    /// </summary>
    /// <param name="ct">CT of the .NET worker service.</param>
    /// <returns>Task for async.</returns>
    public abstract Task Run(CancellationToken ct);
}