using System.Collections.Concurrent;
using SonraML.Backend.MLX.ExecutionManagement;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Core.Exceptions;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.Implementations;

internal class MlxTensorFactory : IGlobalTensorFactory, IScopedTensorFactory
{
    private readonly List<Guid> tensors = [];
    private readonly ConcurrentQueue<MlxCommand> commandQueue;
    private readonly ConcurrentQueue<MlxResponse> responseQueue = new();
    private readonly EventWaitHandle opDone = new(false, EventResetMode.AutoReset);
    private readonly SemaphoreSlim tlock = new(1, 1);

    public MlxTensorFactory(MlxBackendGlobals globals)
    {
        commandQueue = globals.CommandQueue;
    }
    
    private ReplyTo ReplyTo => new(responseQueue, opDone);
    
    public void Dispose()
    {
        commandQueue.Enqueue(new DeleteManyOp(ReplyTo, tensors));
        opDone.WaitOne();
        
        GetResponse<SuccessResponse>();
    }

    public bool IsTypeSupported<T>() where T : struct
    {
        var type = MlxDType.GetDType<T>();

        return type is not null;
    }

    public Tensor<T> Zero<T>(TensorShape shape, string? name = null) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        tlock.Wait();
            
        commandQueue.Enqueue(new CreateZeroOp(ReplyTo, type.Value, shape));
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }

    public Tensor<T> One<T>(TensorShape shape, string? name = null) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        tlock.Wait();
            
        commandQueue.Enqueue(new CreateOneOp(ReplyTo, type.Value, shape));
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }

    public Tensor<T> ScalarZero<T>(string? name = null) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        tlock.Wait();
            
        commandQueue.Enqueue(new CreateScalarZeroOp(ReplyTo, type.Value));
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }

    public Tensor<T> ScalarOne<T>(string? name = null) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        tlock.Wait();
            
        commandQueue.Enqueue(new CreateScalarOneOp(ReplyTo, type.Value));
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }

    public Tensor<T> Create<T>(T scalar, string? name = null) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        tlock.Wait();
            
        if (scalar is bool boolValue)
        {
            commandQueue.Enqueue(new CreateBoolScalarOp(ReplyTo, boolValue));
        }
        
        if (scalar is byte byteValue)
        {
            commandQueue.Enqueue(new CreateU8ScalarOp(ReplyTo, byteValue));
        }
        
        if (scalar is ushort ushortValue)
        {
            commandQueue.Enqueue(new CreateU16ScalarOp(ReplyTo, ushortValue));
        }
        
        if (scalar is uint uintValue)
        {
            commandQueue.Enqueue(new CreateU32ScalarOp(ReplyTo, uintValue));
        }
        
        if (scalar is ulong ulongValue)
        {
            commandQueue.Enqueue(new CreateU64ScalarOp(ReplyTo, ulongValue));
        }
        
        if (scalar is sbyte sbyteValue)
        {
            commandQueue.Enqueue(new CreateI8ScalarOp(ReplyTo, sbyteValue));
        }
        
        if (scalar is short shortValue)
        {
            commandQueue.Enqueue(new CreateI16ScalarOp(ReplyTo, shortValue));
        }
        
        if (scalar is int intValue)
        {
            commandQueue.Enqueue(new CreateI32ScalarOp(ReplyTo, intValue));
        }
        
        if (scalar is long longValue)
        {
            commandQueue.Enqueue(new CreateI64ScalarOp(ReplyTo, longValue));
        }
        
        if (scalar is Half halfValue)
        {
            commandQueue.Enqueue(new CreateF16ScalarOp(ReplyTo, halfValue));
        }
        
        if (scalar is float floatValue)
        {
            commandQueue.Enqueue(new CreateF32ScalarOp(ReplyTo, floatValue));
        }
        
        if (scalar is double doubleValue)
        {
            commandQueue.Enqueue(new CreateF64ScalarOp(ReplyTo, doubleValue));
        }
        
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }

    public Tensor<T> Create<T>(Memory<T> array, TensorShape shape, string? name = null) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }

        if (array.Length != shape.Size)
        {
            throw new BackendOperationException
            (
                $"Array length {array.Length} does not match tensor shape size {shape.Size}."
            );
        }

        tlock.Wait();
            
        if (array is Memory<bool> boolArray)
        {
            commandQueue.Enqueue(new CreateBoolOp(ReplyTo, boolArray, shape));
        }
        
        if (array is Memory<byte> byteArray)
        {
            commandQueue.Enqueue(new CreateU8Op(ReplyTo, byteArray, shape));
        }
        
        if (array is Memory<ushort> ushortArray)
        {
            commandQueue.Enqueue(new CreateU16Op(ReplyTo, ushortArray, shape));
        }
        
        if (array is Memory<uint> uintArray)
        {
            commandQueue.Enqueue(new CreateU32Op(ReplyTo, uintArray, shape));
        }
        
        if (array is Memory<ulong> ulongArray)
        {
            commandQueue.Enqueue(new CreateU64Op(ReplyTo, ulongArray, shape));
        }
        
        if (array is Memory<sbyte> sbyteArray)
        {
            commandQueue.Enqueue(new CreateI8Op(ReplyTo, sbyteArray, shape));
        }
        
        if (array is Memory<short> shortArray)
        {
            commandQueue.Enqueue(new CreateI16Op(ReplyTo, shortArray, shape));
        }
        
        if (array is Memory<int> intArray)
        {
            commandQueue.Enqueue(new CreateI32Op(ReplyTo, intArray, shape));
        }
        
        if (array is Memory<long> longArray)
        {
            commandQueue.Enqueue(new CreateI64Op(ReplyTo, longArray, shape));
        }
        
        if (array is Memory<Half> halfArray)
        {
            commandQueue.Enqueue(new CreateF16Op(ReplyTo, halfArray, shape));
        }
        
        if (array is Memory<float> floatArray)
        {
            commandQueue.Enqueue(new CreateF32Op(ReplyTo, floatArray, shape));
        }
        
        if (array is Memory<double> doubleArray)
        {
            commandQueue.Enqueue(new CreateF64Op(ReplyTo, doubleArray, shape));
        }
        
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }

    public Tensor<T> Arange<T>(double start, double stop, double step, string? name) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        tlock.Wait();
            
        commandQueue.Enqueue(new ArangeOp(ReplyTo, type.Value, start, stop, step));
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }

    public Tensor<T> Linspace<T>(double start, double stop, int samples, string? name) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        tlock.Wait();
            
        commandQueue.Enqueue(new LinspaceOp(ReplyTo, type.Value, start, stop, samples));
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }

    public Tensor<T> Concat<T>(Tensor<T>[] tensors, string? name) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }

        var ids = tensors.Cast<ProxyTensor<T>>().Select(t => t.Id).ToList();
        if (ids.Count != tensors.Length)
        {
            throw new TensorCompatibilityException();
        }
        
        tlock.Wait();
            
        commandQueue.Enqueue(new ConcatOp(ReplyTo, ids));
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }

    public Tensor<T> Concat<T>(Tensor<T>[] tensors, int axis, string? name) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var ids = tensors.Cast<ProxyTensor<T>>().Select(t => t.Id).ToList();
        if (ids.Count != tensors.Length)
        {
            throw new TensorCompatibilityException();
        }
        
        tlock.Wait();
            
        commandQueue.Enqueue(new ConcatAxisOp(ReplyTo, ids, axis));
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }

    public Tensor<T> Stack<T>(Tensor<T>[] tensors, string? name) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var ids = tensors.Cast<ProxyTensor<T>>().Select(t => t.Id).ToList();
        if (ids.Count != tensors.Length)
        {
            throw new TensorCompatibilityException();
        }
        
        tlock.Wait();
            
        commandQueue.Enqueue(new StackOp(ReplyTo, ids));
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }

    public Tensor<T> Stack<T>(Tensor<T>[] tensors, int axis, string? name) where T : struct
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        var ids = tensors.Cast<ProxyTensor<T>>().Select(t => t.Id).ToList();
        if (ids.Count != tensors.Length)
        {
            throw new TensorCompatibilityException();
        }
        
        tlock.Wait();
            
        commandQueue.Enqueue(new StackAxisOp(ReplyTo, ids, axis));
        opDone.WaitOne();
            
        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return CreateProxyTensor<T>(tr.Id, tr.Type, name);
    }
    
    public ProxyTensor<TTarget> CreateProxyTensor<TTarget>(Guid id, DType type, string? name = null) where TTarget : struct
    {
        var t = MlxDType.GetType(type);
        if (t is null || t != typeof(TTarget))
        {
            throw new BackendOperationException
                ($"Type missmatch from MLX. Expected '{typeof(TTarget).Name}' but got '{t?.Name}'.");
        }

        return type switch
        {
            DType.Bool => (ProxyTensor<TTarget>)(object)AddTensor<bool>(id, name),
            DType.UInt8 => (ProxyTensor<TTarget>)(object)AddTensor<byte>(id, name),
            DType.UInt16 => (ProxyTensor<TTarget>)(object)AddTensor<ushort>(id, name),
            DType.UInt32 => (ProxyTensor<TTarget>)(object)AddTensor<uint>(id, name),
            DType.UInt64 => (ProxyTensor<TTarget>)(object)AddTensor<ulong>(id, name),
            DType.Int8 => (ProxyTensor<TTarget>)(object)AddTensor<sbyte>(id, name),
            DType.Int16 => (ProxyTensor<TTarget>)(object)AddTensor<short>(id, name),
            DType.Int32 => (ProxyTensor<TTarget>)(object)AddTensor<int>(id, name),
            DType.Int64 => (ProxyTensor<TTarget>)(object)AddTensor<long>(id, name),
            DType.Float16 => (ProxyTensor<TTarget>)(object)AddTensor<Half>(id, name),
            DType.Float32 => (ProxyTensor<TTarget>)(object)AddTensor<float>(id, name),
            DType.Float64 => (ProxyTensor<TTarget>)(object)AddTensor<double>(id, name),
            _ => throw new ArgumentOutOfRangeException(nameof(type)),
        };
    }

    private ProxyTensor<T> AddTensor<T>(Guid id, string? name) where T : struct
    {
        tensors.Add(id);

        return new ProxyTensor<T>(id, this, commandQueue, name);
    }
    
    private TResponse GetResponse<TResponse>() where TResponse : MlxResponse
    {
        if (!responseQueue.TryDequeue(out var response))
        {
            tlock.Release();
            throw new BackendOperationException("Expected MLX Response.");
        }
        
        if (response is ErrorResponse error)
        {
            tlock.Release();
            throw new BackendOperationException(error.Message);
        }
        
        if (response is not TResponse tr)
        {
            tlock.Release();
            throw new BackendOperationException($"Expected {typeof(TResponse).Name}.");
        }

        return tr;
    }
}