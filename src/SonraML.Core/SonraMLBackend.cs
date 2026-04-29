using SonraML.Core.Interfaces;

namespace SonraML.Core;

public abstract class SonraMLBackend : IDisposable
{
    public ITensorFactory TensorFactory { get;  protected set; }
    
    public abstract void Dispose();
}