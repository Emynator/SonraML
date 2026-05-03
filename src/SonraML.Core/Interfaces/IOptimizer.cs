using SonraML.Core.Types;

namespace SonraML.Core.Interfaces;

/// <summary>
/// Interface that all optimizers have to implement.
/// </summary>
/// <typeparam name="T">Type of the tensor.</typeparam>
public interface IOptimizer<T> where T : struct
{
    public void Step(IEnumerable<Parameter<T>> parameters);
}