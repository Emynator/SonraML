using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX;

internal class MlxTensorFactory : ITensorFactory
{
    public bool IsTypeSupported<T>()
    {
        return MlxDType.GetDType<T>() is not null;
    }

    public Tensor<T> Zero<T>(TensorShape shape, string? name = null) where T : struct
    {
        var result = new MlxTensor<T>(shape, name);
        result.SetZero();
        
        return result;
    }

    public Tensor<T> One<T>(TensorShape shape, string? name = null) where T : struct
    {
        var result = new MlxTensor<T>(shape, name);
        result.SetOne();
        
        return result;
    }

    public Tensor<T> Create<T>(T scalar, string? name = null) where T : struct
    {
        return new MlxTensor<T>(scalar, name);
    }

    public Tensor<T> Create<T>(Memory<T> array, TensorShape shape, string? name = null) where T : struct
    {
        return new MlxTensor<T>(array, shape, name);
    }
}