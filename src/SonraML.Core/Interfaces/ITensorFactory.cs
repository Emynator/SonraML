using SonraML.Core.Types;

namespace SonraML.Core.Interfaces;

public interface ITensorFactory
{
    public bool IsTypeSupported(Type t);
    
    public Tensor<T> Zero<T>(TensorShape shape) where T : struct;

    public Tensor<T> One<T>(TensorShape shape) where T : struct;
    
    public Tensor<T> Create<T>(T scalar) where T : struct;
    
    public Tensor<T> Create<T>(Span<T> array, TensorShape shape) where T : struct;
}