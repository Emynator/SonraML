namespace SonraML.Core.Exceptions;

public class RunnerNotFoundException : SonraMLException
{
    public RunnerNotFoundException(string name) : base($"Runner '{name}' not found.")
    {
    }
}