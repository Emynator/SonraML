using Microsoft.Extensions.Logging;
using SonraML.Backend.MLX.Extensions;
using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Backend.MLX.Managed;
using SonraML.Core.Enums;
using SonraML.Core.Exceptions;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.ExecutionManagement;

internal unsafe class MlxTensorManager : IDisposable
{
    private readonly ILogger<MlxTensorManager> logger;
    private readonly MlxBackendGlobals globals;
    private readonly Dictionary<Guid, MlxTensor> tensors = [];
    private ManagedMlxStream? stream;

    public MlxTensorManager(ILogger<MlxTensorManager> logger, MlxBackendGlobals globals)
    {
        this.logger = logger;
        this.globals = globals;
    }

    private ManagedMlxStream Stream => stream ?? throw new BackendNotInitializedException();

    public void Dispose()
    {
        foreach (var tensor in tensors)
        {
            tensor.Value.Dispose();
        }

        Stream?.Dispose();
    }

    public void Init()
    {
        if (stream is not null)
        {
            stream.Dispose();
            stream = null;
        }
        
        var deviceType = globals.DeviceType switch
        {
            BackendDeviceType.Cpu => MlxDeviceType.Cpu,
            BackendDeviceType.Gpu => MlxDeviceType.Gpu,
            _ => throw new BackendOperationException("Invalid backend device."),
        };

        stream = new ManagedMlxStream(deviceType);
    }

    public MlxTensor? Get(Guid id)
    {
        tensors.TryGetValue(id, out var result);
        return result;
    }

    public MlxTensor[] Get(List<Guid> ids)
    {
        return tensors
            .Where(t => ids.Contains(t.Key))
            .Select(t => t.Value)
            .ToArray();
    }

    public void Delete(Guid id)
    {
        tensors.TryGetValue(id, out var tensor);
        if (tensor is not null)
        {
            tensors.Remove(id);
            tensor.Dispose();
        }
    }

    public void Delete(List<Guid> ids)
    {
        var toDelete = tensors
            .Where(t => ids.Contains(t.Key))
            .Select(t => t.Value);

        foreach (var tensor in toDelete)
        {
            tensors.Remove(tensor.Id);
            tensor.Dispose();
        }
    }

    public MlxTensor Zero(DType type, TensorShape shape)
    {
        var result = new MlxTensor(this, Stream.Stream, shape);
        tensors.Add(result.Id, result);

        using var handle = shape.GetHandle();
        MlxOps.Zeros
        (
            in result.Array.Array,
            (int*)handle.Pointer,
            (UIntPtr)shape.Dimensions,
            type,
            Stream.Stream
        );

        return result;
    }

    public MlxTensor One(DType type, TensorShape shape)
    {
        var result = new MlxTensor(this, Stream.Stream, shape);
        tensors.Add(result.Id, result);

        using var shapeHandle = shape.GetHandle();
        MlxOps.Ones
        (
            in result.Array.Array,
            (int*)shapeHandle.Pointer,
            (UIntPtr)shape.Dimensions,
            type,
            Stream.Stream
        );

        return result;
    }

    public MlxTensor ScalarZero(DType type)
    {
        var result = type switch
        {
            DType.Bool => new MlxTensor(this, Stream.Stream, false),
            DType.UInt8 => new MlxTensor(this, Stream.Stream, (byte)0),
            DType.UInt16 => new MlxTensor(this, Stream.Stream, (ushort)0),
            DType.UInt32 => new MlxTensor(this, Stream.Stream, (uint)0),
            DType.UInt64 => new MlxTensor(this, Stream.Stream, (ulong)0),
            DType.Int8 => new MlxTensor(this, Stream.Stream, (sbyte)0),
            DType.Int16 => new MlxTensor(this, Stream.Stream, (short)0),
            DType.Int32 => new MlxTensor(this, Stream.Stream, 0),
            DType.Int64 => new MlxTensor(this, Stream.Stream, (long)0),
            DType.Float16 => new MlxTensor(this, Stream.Stream, (Half)0.0f),
            DType.Float32 => new MlxTensor(this, Stream.Stream, 0.0f),
            DType.Float64 => new MlxTensor(this, Stream.Stream, 0.0d),
            _ => throw new ArgumentOutOfRangeException(nameof(type)),
        };
        tensors.Add(result.Id, result);

        return result;
    }

    public MlxTensor ScalarOne(DType type)
    {
        var result = type switch
        {
            DType.Bool => new MlxTensor(this, Stream.Stream, true),
            DType.UInt8 => new MlxTensor(this, Stream.Stream, (byte)1),
            DType.UInt16 => new MlxTensor(this, Stream.Stream, (ushort)1),
            DType.UInt32 => new MlxTensor(this, Stream.Stream, (uint)1),
            DType.UInt64 => new MlxTensor(this, Stream.Stream, (ulong)1),
            DType.Int8 => new MlxTensor(this, Stream.Stream, (sbyte)1),
            DType.Int16 => new MlxTensor(this, Stream.Stream, (short)1),
            DType.Int32 => new MlxTensor(this, Stream.Stream, 1),
            DType.Int64 => new MlxTensor(this, Stream.Stream, (long)1),
            DType.Float16 => new MlxTensor(this, Stream.Stream, (Half)1.0f),
            DType.Float32 => new MlxTensor(this, Stream.Stream, 1.0f),
            DType.Float64 => new MlxTensor(this, Stream.Stream, 1.0d),
            _ => throw new ArgumentOutOfRangeException(nameof(type)),
        };
        tensors.Add(result.Id, result);

        return result;
    }

    public MlxTensor Create<T>(T scalar) where T : struct
    {
        if (scalar is bool boolVal)
        {
            var result = new MlxTensor(this, Stream.Stream, boolVal);
            tensors.Add(result.Id, result);

            return result;
        }

        if (scalar is byte u8Val)
        {
            var result = new MlxTensor(this, Stream.Stream, u8Val);
            tensors.Add(result.Id, result);

            return result;
        }

        if (scalar is ushort u16Val)
        {
            var result = new MlxTensor(this, Stream.Stream, u16Val);
            tensors.Add(result.Id, result);

            return result;
        }

        if (scalar is uint u32Val)
        {
            var result = new MlxTensor(this, Stream.Stream, u32Val);
            tensors.Add(result.Id, result);

            return result;
        }

        if (scalar is ulong u64Val)
        {
            var result = new MlxTensor(this, Stream.Stream, u64Val);
            tensors.Add(result.Id, result);

            return result;
        }

        if (scalar is sbyte i8Val)
        {
            var result = new MlxTensor(this, Stream.Stream, i8Val);
            tensors.Add(result.Id, result);

            return result;
        }

        if (scalar is short i16Val)
        {
            var result = new MlxTensor(this, Stream.Stream, i16Val);
            tensors.Add(result.Id, result);

            return result;
        }

        if (scalar is int i32Val)
        {
            var result = new MlxTensor(this, Stream.Stream, i32Val);
            tensors.Add(result.Id, result);

            return result;
        }

        if (scalar is long i64Val)
        {
            var result = new MlxTensor(this, Stream.Stream, i64Val);
            tensors.Add(result.Id, result);

            return result;
        }

        if (scalar is Half f16Val)
        {
            var result = new MlxTensor(this, Stream.Stream, f16Val);
            tensors.Add(result.Id, result);

            return result;
        }

        if (scalar is float f32Val)
        {
            var result = new MlxTensor(this, Stream.Stream, f32Val);
            tensors.Add(result.Id, result);

            return result;
        }

        if (scalar is double f64Val)
        {
            var result = new MlxTensor(this, Stream.Stream, f64Val);
            tensors.Add(result.Id, result);

            return result;
        }
        
        throw new TensorTypeNotSupportedException(typeof(T));
    }

    public MlxTensor Create<T>(Memory<T> array, TensorShape shape) where T : struct
    {
        if (array is Memory<bool> boolMemory)
        {
            var result = new MlxTensor(this, Stream.Stream, boolMemory, shape);
            tensors.Add(result.Id, result);

            return result;
        }

        if (array is Memory<byte> u8Memory)
        {
            var result = new MlxTensor(this, Stream.Stream, u8Memory, shape);
            tensors.Add(result.Id, result);

            return result;
        }

        if (array is Memory<ushort> u16Memory)
        {
            var result = new MlxTensor(this, Stream.Stream, u16Memory, shape);
            tensors.Add(result.Id, result);

            return result;
        }

        if (array is Memory<uint> u32Memory)
        {
            var result = new MlxTensor(this, Stream.Stream, u32Memory, shape);
            tensors.Add(result.Id, result);

            return result;
        }

        if (array is Memory<ulong> u64Memory)
        {
            var result = new MlxTensor(this, Stream.Stream, u64Memory, shape);
            tensors.Add(result.Id, result);

            return result;
        }

        if (array is Memory<sbyte> i8Memory)
        {
            var result = new MlxTensor(this, Stream.Stream, i8Memory, shape);
            tensors.Add(result.Id, result);

            return result;
        }

        if (array is Memory<short> i16Memory)
        {
            var result = new MlxTensor(this, Stream.Stream, i16Memory, shape);
            tensors.Add(result.Id, result);

            return result;
        }

        if (array is Memory<int> i32Memory)
        {
            var result = new MlxTensor(this, Stream.Stream, i32Memory, shape);
            tensors.Add(result.Id, result);

            return result;
        }

        if (array is Memory<long> i64Memory)
        {
            var result = new MlxTensor(this, Stream.Stream, i64Memory, shape);
            tensors.Add(result.Id, result);

            return result;
        }

        if (array is Memory<Half> f16Memory)
        {
            var result = new MlxTensor(this, Stream.Stream, f16Memory, shape);
            tensors.Add(result.Id, result);

            return result;
        }

        if (array is Memory<float> f32Memory)
        {
            var result = new MlxTensor(this, Stream.Stream, f32Memory, shape);
            tensors.Add(result.Id, result);

            return result;
        }

        if (array is Memory<double> f64Memory)
        {
            var result = new MlxTensor(this, Stream.Stream, f64Memory, shape);
            tensors.Add(result.Id, result);

            return result;
        }
        
        throw new TensorTypeNotSupportedException(typeof(T));
    }

    public MlxTensor Arange(DType type, double start, double stop, double step)
    {
        var result = new MlxTensor(this, Stream.Stream);
        tensors.Add(result.Id, result);
        
        MlxOps.Arange(in result.Array.Array, start, stop, step, type, Stream.Stream);

        return result;
    }

    public MlxTensor Linspace(DType type, double start, double stop, int samples)
    {
        var result = new MlxTensor(this, Stream.Stream);
        tensors.Add(result.Id, result);
        
        MlxOps.Linspace(in result.Array.Array, start, stop, samples, type, Stream.Stream);

        return result;
    }

    public MlxTensor Concat(MlxTensor[] tensors)
    {
        var result = new MlxTensor(this, Stream.Stream);
        this.tensors.Add(result.Id, result);

        using var vec = new ManagedMlxVectorArray(tensors);
        MlxOps.Concatenate(in result.Array.Array, vec.Vector, Stream.Stream);

        return result;
    }

    public MlxTensor Concat(MlxTensor[] tensors, int axis)
    {
        var result = new MlxTensor(this, Stream.Stream);
        this.tensors.Add(result.Id, result);

        using var vec = new ManagedMlxVectorArray(tensors);
        MlxOps.ConcatenateAxis(in result.Array.Array, vec.Vector, axis, Stream.Stream);

        return result;
    }

    public MlxTensor Stack(MlxTensor[] tensors)
    {
        var result = new MlxTensor(this, Stream.Stream);
        this.tensors.Add(result.Id, result);

        using var vec = new ManagedMlxVectorArray(tensors);
        MlxOps.Stack(in result.Array.Array, vec.Vector, Stream.Stream);

        return result;
    }

    public MlxTensor Stack(MlxTensor[] tensors, int axis)
    {
        var result = new MlxTensor(this, Stream.Stream);
        this.tensors.Add(result.Id, result);

        using var vec = new ManagedMlxVectorArray(tensors);
        MlxOps.StackAxis(in result.Array.Array, vec.Vector, axis, Stream.Stream);

        return result;
    }

    public MlxTensor CreateEmpty()
    {
        var result = new MlxTensor(this, Stream.Stream);
        tensors.Add(result.Id, result);

        return result;
    }
}