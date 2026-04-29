using SonraML.Core.Types;

namespace SonraML.Core.Interfaces;

public interface ITensorReader
{
    public ICollection<GenericTensor> ReadTensors();

    public GenericTensor? ReadTensor(string name);
}