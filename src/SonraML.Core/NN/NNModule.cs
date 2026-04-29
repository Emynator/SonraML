using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public abstract class NNModule<T> : IDisposable where T : struct
{
    private readonly string name;

    public NNModule(string? name = null)
    {
        if (string.IsNullOrEmpty(name))
        {
            name = Guid.NewGuid().ToString();
        }
    }

    protected ITensorFactory Tf => SonraMLConfiguration.Backend.TensorFactory;

    public abstract void Dispose();

    public abstract void Forward(Tensor<T> x);
}