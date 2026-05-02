using Microsoft.Extensions.Logging;
using SonraML.Backend.MLX.Interfaces;
using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Backend.MLX.Managed;
using SonraML.Core.Enums;
using SonraML.Core.Exceptions;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.Implementations;

internal class MlxTensorFactory : IGlobalTensorFactory, IScopedTensorFactory
{
    private readonly ILogger<MlxTensorFactory> logger;

    [ThreadStatic]
    private static ManagedMlxStream? stream;

    private readonly BackendDeviceType deviceType;
    private readonly List<ManagedMlxStream> streams = [];
    private readonly List<GenericTensor> tensors = [];

    public MlxTensorFactory(ILogger<MlxTensorFactory> logger, IMlxBackendGlobals globals)
    {
        this.logger = logger;
        deviceType = globals.DeviceType;
    }

    public ManagedMlxStream Stream
    {
        get
        {
            logger.LogDebug("Stream access from thread '{thread}'", Thread.CurrentThread.ManagedThreadId);
            if (stream is null)
            {
                logger.LogDebug("Thread '{thread}' created new stream.", Thread.CurrentThread.ManagedThreadId);
                stream = new ManagedMlxStream(deviceType);
                streams.Add(stream);
            }
            
            return stream;
        }
    }

    public void Dispose()
    {
        foreach (var tensor in tensors)
        {
            tensor.Release();
        }

        foreach (var s in streams)
        {
            s.Dispose();
        }
    }

    public bool IsTypeSupported<T>()
    {
        return MlxDType.GetDType<T>() is not null;
    }

    public Tensor<T> Zero<T>(TensorShape shape, string? name = null) where T : struct
    {
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var result = new MlxTensor<T>(this, shape, name);
        tensors.Add(result);
        
        result.SetZero();
        
        return result;
    }

    public Tensor<T> One<T>(TensorShape shape, string? name = null) where T : struct
    {
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var result = new MlxTensor<T>(this, shape, name);
        tensors.Add(result);
        
        result.SetOne();
        
        return result;
    }

    public Tensor<T> ScalarZero<T>(string? name = null) where T : struct
    {
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var result = new MlxTensor<T>(this, true, name);
        tensors.Add(result);

        return result;
    }

    public Tensor<T> ScalarOne<T>(string? name = null) where T : struct
    {
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var result = new MlxTensor<T>(this, false, name);
        tensors.Add(result);
        
        return result;
    }

    public Tensor<T> Create<T>(T scalar, string? name = null) where T : struct
    {
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var result = new MlxTensor<T>(this, scalar, name);
        tensors.Add(result);
        
        return result;
    }

    public Tensor<T> Create<T>(Memory<T> array, TensorShape shape, string? name = null) where T : struct
    {
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var result = new MlxTensor<T>(this, array, shape, name);
        tensors.Add(result);
        
        return result;
    }

    public Tensor<T> Arange<T>(double start, double stop, double step, string? name) where T : struct
    {
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var result = new MlxTensor<T>(this, name);
        tensors.Add(result);
        MlxOps.Arange(in result.Array.Array, start, stop, step, dtype.Value, Stream.Stream);
        
        return result;
    }

    public Tensor<T> Linspace<T>(double start, double stop, int samples, string? name) where T : struct
    {
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var result = new MlxTensor<T>(this, name);
        tensors.Add(result);
        MlxOps.Linspace(in result.Array.Array, start, stop, samples, dtype.Value, Stream.Stream);
        
        return result;
    }
    
    public Tensor<T> Concat<T>(Tensor<T>[] tensors, string? name) where T : struct
    {
        if (tensors is not MlxTensor<T>[] ts)
        {
            throw new TensorCompatibilityException();
        }
        
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }

        var result = new MlxTensor<T>(this, name);
        this.tensors.Add(result);
        
        using var vec = new ManagedMlxVectorArray<T>(ts);
        MlxOps.Concatenate(in result.Array.Array, vec.Vector, Stream.Stream);
        
        return result;
    }

    public Tensor<T> Concat<T>(Tensor<T>[] tensors, int axis, string? name) where T : struct
    {
        if (tensors is not MlxTensor<T>[] ts)
        {
            throw new TensorCompatibilityException();
        }
        
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var result = new MlxTensor<T>(this, name);
        this.tensors.Add(result);
        
        using var vec = new ManagedMlxVectorArray<T>(ts);
        MlxOps.ConcatenateAxis(in result.Array.Array, vec.Vector, axis, Stream.Stream);
        
        return result;
    }

    public Tensor<T> Stack<T>(Tensor<T>[] tensors, string? name) where T : struct
    {
        if (tensors is not MlxTensor<T>[] ts)
        {
            throw new TensorCompatibilityException();
        }
        
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var result = new MlxTensor<T>(this, name);
        this.tensors.Add(result);
        
        using var vec = new ManagedMlxVectorArray<T>(ts);
        MlxOps.Stack(in result.Array.Array, vec.Vector, Stream.Stream);
        
        return result;
    }

    public Tensor<T> Stack<T>(Tensor<T>[] tensors, int axis, string? name) where T : struct
    {
        var ts = tensors.Select(t => t as MlxTensor<T>).ToArray();
        if (ts is null || ts.Length != tensors.Length)
        {
            throw new TensorCompatibilityException();
        }
        
        var dtype = MlxDType.GetDType<T>();
        if (dtype is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var result = new MlxTensor<T>(this, name);
        this.tensors.Add(result);
        
        using var vec = new ManagedMlxVectorArray<T>(ts);
        MlxOps.StackAxis(in result.Array.Array, vec.Vector, axis, Stream.Stream);
        
        return result;
    }

    internal MlxTensor<T> CreateEmpty<T>(string? name = null) where T : struct
    {
        var result = new MlxTensor<T>(this, name);
        tensors.Add(result);
        
        return result;
    }
}