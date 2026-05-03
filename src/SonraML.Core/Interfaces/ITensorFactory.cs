using SonraML.Core.Types;

namespace SonraML.Core.Interfaces;

public interface ITensorFactory : IDisposable
{
    public bool IsTypeSupported<T>() where T : struct;
    
    public Tensor<T> Zero<T>(TensorShape shape, string? name = null) where T : struct;

    public Tensor<T> One<T>(TensorShape shape, string? name = null) where T : struct;
    
    public Tensor<T> ScalarZero<T>(string? name = null) where T : struct;
    
    public Tensor<T> ScalarOne<T>(string? name = null) where T : struct;
    
    public Tensor<T> Create<T>(T scalar, string? name = null) where T : struct;
    
    public Tensor<T> Create<T>(Memory<T> array, TensorShape shape, string? name = null) where T : struct;

    public Tensor<T> Arange<T>(double start, double stop, double step, string? name) where T : struct;

    public Tensor<T> Linspace<T>(double start, double stop, int samples, string? name)  where T : struct;

    public Tensor<T> Concat<T>(Tensor<T>[] tensors, string? name) where T : struct;

    public Tensor<T> Concat<T>(Tensor<T>[] tensors, int axis, string? name) where T : struct;

    public Tensor<T> Stack<T>(Tensor<T>[] tensors, string? name) where T : struct;

    public Tensor<T> Stack<T>(Tensor<T>[] tensors, int axis, string? name) where T : struct;
}

public interface IGlobalTensorFactory : ITensorFactory;

public interface IScopedTensorFactory : ITensorFactory;