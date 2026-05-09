using SonraML.Core.Interfaces;
using SonraML.Core.Services;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public abstract class NNContainer<T> : NNModule<T> where T : struct
{
    public NNContainer(ModuleFactory factory)
    {
        Factory = factory;
    }
    
    public ModuleFactory Factory { get; }

    public abstract void AddModule(NNModule<T> module);
}