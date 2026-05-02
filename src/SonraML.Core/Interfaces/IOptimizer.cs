using SonraML.Core.Types;

namespace SonraML.Core.Interfaces;

public interface IOptimizer<T> where T : struct
{
    public void Step(IEnumerable<Parameter<T>> parameters);
}