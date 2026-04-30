using SonraML.Backend.MLX.Interfaces;
using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Core.Exceptions;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.Implementations;

internal class MlxTensorFactory : IGlobalTensorFactory, IScopedTensorFactory
{
    private readonly MlxStream stream;
    private readonly List<GenericTensor> tensors = [];

    public MlxTensorFactory(IMlxBackendGlobals globals)
    {
        stream = globals.Stream?.Stream ?? throw new BackendNotInitializedException();
    }
    
    public void Dispose()
    {
        foreach (var tensor in tensors)
        {
            tensor.Release();
        }
    }

    public bool IsTypeSupported<T>()
    {
        return MlxDType.GetDType<T>() is not null;
    }

    public Tensor<T> Zero<T>(TensorShape shape, string? name = null) where T : struct
    {
        var result = new MlxTensor<T>(this, stream, shape, name);
        tensors.Add(result);
        
        result.SetZero();
        
        return result;
    }

    public Tensor<T> One<T>(TensorShape shape, string? name = null) where T : struct
    {
        var result = new MlxTensor<T>(this, stream, shape, name);
        tensors.Add(result);
        
        result.SetOne();
        
        return result;
    }

    public Tensor<T> ScalarZero<T>(string? name = null) where T : struct
    {
        var result = new MlxTensor<T>(this, stream, true, name);
        tensors.Add(result);

        return result;
    }

    public Tensor<T> ScalarOne<T>(string? name = null) where T : struct
    {
        var result = new MlxTensor<T>(this, stream, false, name);
        tensors.Add(result);
        
        return result;
    }

    public Tensor<T> Create<T>(T scalar, string? name = null) where T : struct
    {
        var result = new MlxTensor<T>(this, stream, scalar, name);
        tensors.Add(result);
        
        return result;
    }

    public Tensor<T> Create<T>(Memory<T> array, TensorShape shape, string? name = null) where T : struct
    {
        var result = new MlxTensor<T>(this, stream, array, shape, name);
        tensors.Add(result);
        
        return result;
    }

    internal MlxTensor<T> CreateEmpty<T>(string? name = null) where T : struct
    {
        var result = new MlxTensor<T>(this, stream, name);
        tensors.Add(result);
        
        return result;
    }
}