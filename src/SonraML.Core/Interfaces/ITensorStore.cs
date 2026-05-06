using SonraML.Core.Types;

namespace SonraML.Core.Interfaces;

public interface ITensorStore
{
    public Task<ICollection<string>> ListTensors();

    public Task<bool> Contains(string key);
    
    public Task<bool> Contains(ICollection<string> keys);

    public Task<ICollection<GenericTensor>> LoadTensors();

    public Task<ICollection<GenericTensor>> LoadTensors(ICollection<string> keys);
    
    public Task<GenericTensor?> LoadTensor(string key);
    
    public Task AddTensor(GenericTensor tensor);

    public Task AddTensors(ICollection<GenericTensor> tensors);

    public Task RemoveTensor(string key);

    public Task Persist();
}