using SonraML.Core.Types;

namespace SonraML.Core.Interfaces;

public interface ITensorFactory
{
    public bool IsTypeSupported<T>();
    
    public Tensor<T> Zero<T>(TensorShape shape, string? name = null) where T : struct;

    public Tensor<T> One<T>(TensorShape shape, string? name = null) where T : struct;
    
    public Tensor<T> ScalarZero<T>(string? name = null) where T : struct;
    
    public Tensor<T> ScalarOne<T>(string? name = null) where T : struct;
    
    public Tensor<T> Create<T>(T scalar, string? name = null) where T : struct;
    
    public Tensor<T> Create<T>(Memory<T> array, TensorShape shape, string? name = null) where T : struct;
}