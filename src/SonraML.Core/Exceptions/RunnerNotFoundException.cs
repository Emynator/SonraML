namespace SonraML.Core.Exceptions;

public sealed class RunnerNotFoundException : SonraMLException
{
    public RunnerNotFoundException(string name) : base($"Runner '{name}' not found.")
    {
    }
}