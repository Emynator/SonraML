using SonraML.Core.Types;

namespace SonraML.Core.Interfaces;

public interface ITensorWriter
{
    public void WriteTensor(GenericTensor tensor);

    public void WriteTensors(ICollection<GenericTensor> tensors);
}