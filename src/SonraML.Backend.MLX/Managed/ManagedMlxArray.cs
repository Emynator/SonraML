using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Core.Exceptions;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.Managed;

internal unsafe class ManagedMlxArray<T> : IDisposable where T : struct
{
    public MlxArray Array;
    
    public ManagedMlxArray()
    {
        Array = MlxArray.New();
    }
    
    public ManagedMlxArray(Memory<T> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        
        if (typeof(T) == typeof(bool))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Bool);
            return;
        }

        if (typeof(T) == typeof(byte))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.UInt8);
            return;
        }

        if (typeof(T) == typeof(ushort))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.UInt16);
            return;
        }

        if (typeof(T) == typeof(uint))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.UInt32);
            return;
        }

        if (typeof(T) == typeof(ulong))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.UInt64);
            return;
        }

        if (typeof(T) == typeof(sbyte))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Int8);
            return;
        }

        if (typeof(T) == typeof(short))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Int16);
            return;
        }

        if (typeof(T) == typeof(int))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Int32);
            return;
        }

        if (typeof(T) == typeof(long))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Int64);
            return;
        }

        if (typeof(T) == typeof(Half))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Float16);
            return;
        }

        if (typeof(T) == typeof(float))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Float32);
            return;
        }

        if (typeof(T) == typeof(double))
        {
            MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Float64);
            return;
        }
        
        throw new TensorTypeNotSupportedException(typeof(T));
    }

    public ManagedMlxArray(T scalar)
    {
        if (scalar is bool boolValue)
        {
            Array = MlxArray.NewBool(boolValue);
            return;
        }

        if (scalar is byte byteValue)
        {
            Array = MlxArray.NewData(&byteValue, null, 0, DType.UInt8);
            return;
        }

        if (scalar is ushort ushortValue)
        {
            Array = MlxArray.NewData(&ushortValue, null, 0, DType.UInt16);
            return;
        }

        if (scalar is uint uintValue)
        {
            Array = MlxArray.NewData(&uintValue, null, 0, DType.UInt32);
            return;
        }

        if (scalar is ulong ulongValue)
        {
            Array = MlxArray.NewData(&ulongValue, null, 0, DType.UInt64);
            return;
        }

        if (scalar is sbyte sbyteValue)
        {
            Array = MlxArray.NewData(&sbyteValue, null, 0, DType.Int8);
            return;
        }

        if (scalar is short shortValue)
        {
            Array = MlxArray.NewData(&shortValue, null, 0, DType.Int16);
            return;
        }
        
        if (scalar is int intValue)
        {
            Array = MlxArray.NewInt(intValue);
            return;
        }

        if (scalar is long longValue)
        {
            Array = MlxArray.NewData(&longValue, null, 0, DType.Int64);
            return;
        }

        if (scalar is Half halfValue)
        {
            Array = MlxArray.NewData(&halfValue, null, 0, DType.Float16);
            return;
        }

        if (scalar is float floatValue)
        {
            Array = MlxArray.NewFloat32(floatValue);
            return;
        }

        if (scalar is double doubleValue)
        {
            Array = MlxArray.NewFloat64(doubleValue);
            return;
        }
        
        throw new TensorTypeNotSupportedException(typeof(T));
    }

    public void Dispose()
    {
        MlxArray.Free(Array);
    }

    public override string ToString()
    {
        MlxArray.ToString(out var mlxString, Array);
        var result = MlxString.Data(mlxString);
        return result;
    }

    public void CopyFrom(ManagedMlxArray<T> array)
    {
        MlxArray.Set(in Array, array.Array);
    }

    public TensorShape GetShape()
    {
        Eval();
        
        var shape = MlxArray.Shape(Array);
        var length = MlxArray.NDim(Array);
        var result = new int[length];
        for (UIntPtr i = 0; i < length; i++)
        {
            result[i] = shape[i];
        }
        
        return new TensorShape(result);
    }

    public T GetScalar()
    {
        Eval();
        
        if (typeof(T) == typeof(bool))
        {
            MlxArray.ItemBool(out var res, Array);
            return (T)(object)res;
        }

        if (typeof(T) == typeof(byte))
        {
            MlxArray.ItemUInt8(out var res, Array);
            return (T)(object)res;
        }

        if (typeof(T) == typeof(ushort))
        {
            MlxArray.ItemUInt16(out var res, Array);
            return (T)(object)res;
        }

        if (typeof(T) == typeof(uint))
        {
            MlxArray.ItemUInt32(out var res, Array);
            return (T)(object)res;
        }

        if (typeof(T) == typeof(ulong))
        {
            MlxArray.ItemUInt64(out var res, Array);
            return (T)(object)res;
        }

        if (typeof(T) == typeof(sbyte))
        {
            MlxArray.ItemInt8(out var res, Array);
            return (T)(object)res;
        }

        if (typeof(T) == typeof(short))
        {
            MlxArray.ItemInt16(out var res, Array);
            return (T)(object)res;
        }

        if (typeof(T) == typeof(int))
        {
            MlxArray.ItemInt32(out var res, Array);
            return (T)(object)res;
        }

        if (typeof(T) == typeof(long))
        {
            MlxArray.ItemInt64(out var res, Array);
            return (T)(object)res;
        }

        if (typeof(T) == typeof(Half))
        {
            MlxArray.ItemFloat16(out var res, Array);
            return (T)(object)res;
        }

        if (typeof(T) == typeof(float))
        {
            MlxArray.ItemFloat32(out var res, Array);
            return (T)(object)res;
        }

        if (typeof(T) == typeof(double))
        {
            MlxArray.ItemFloat64(out var res, Array);
            return (T)(object)res;
        }
        
        throw new TensorTypeNotSupportedException(typeof(T));
    }

    public T[] GetData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        
        if (typeof(T) == typeof(bool))
        {
            bool[] res = new bool[size];
            var ptr = MlxArray.DataBool(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i] != 0;
            }
            
            return (T[])(object)res;
        }

        if (typeof(T) == typeof(byte))
        {
            byte[] res = new byte[size];
            var ptr = MlxArray.DataUInt8(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i];
            }
            
            return (T[])(object)res;
        }

        if (typeof(T) == typeof(ushort))
        {
            ushort[] res = new ushort[size];
            var ptr = MlxArray.DataUInt16(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i];
            }
            
            return (T[])(object)res;
        }

        if (typeof(T) == typeof(uint))
        {
            uint[] res = new uint[size];
            var ptr = MlxArray.DataUInt32(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i];
            }
            
            return (T[])(object)res;
        }

        if (typeof(T) == typeof(ulong))
        {
            ulong[] res = new ulong[size];
            var ptr = MlxArray.DataUInt64(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i];
            }
            
            return (T[])(object)res;
        }

        if (typeof(T) == typeof(sbyte))
        {
            sbyte[] res = new sbyte[size];
            var ptr = MlxArray.DataInt8(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i];
            }
            
            return (T[])(object)res;
        }

        if (typeof(T) == typeof(short))
        {
            short[] res = new short[size];
            var ptr = MlxArray.DataInt16(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i];
            }
            
            return (T[])(object)res;
        }

        if (typeof(T) == typeof(int))
        {
            int[] res = new int[size];
            var ptr = MlxArray.DataInt32(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i];
            }
            
            return (T[])(object)res;
        }

        if (typeof(T) == typeof(long))
        {
            long[] res = new long[size];
            var ptr = MlxArray.DataInt64(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i];
            }
            
            return (T[])(object)res;
        }

        if (typeof(T) == typeof(Half))
        {
            Half[] res = new Half[size];
            var ptr = (Half*)MlxArray.DataFloat16(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i];
            }
            
            return (T[])(object)res;
        }

        if (typeof(T) == typeof(float))
        {
            float[] res = new float[size];
            var ptr = MlxArray.DataFloat32(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i];
            }
            
            return (T[])(object)res;
        }

        if (typeof(T) == typeof(double))
        {
            double[] res = new double[size];
            var ptr = MlxArray.DataFloat64(Array);
            for (UIntPtr i = 0; i < size; i++)
            {
                res[i] = ptr[i];
            }
            
            return (T[])(object)res;
        }
        
        throw new TensorTypeNotSupportedException(typeof(T));
    }

    public IEnumerator<T> GetEnumerator()
    {
        Eval();
        var size = MlxArray.Size(Array);
        
        if (typeof(T) == typeof(bool))
        {
            var ptr = MlxArray.DataBool(Array);
            
            return new MlxArrayBoolEnumerator(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }

        if (typeof(T) == typeof(byte))
        {
            var ptr = MlxArray.DataUInt8(Array);
            
            return new MlxArrayEnumerator<byte>(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }

        if (typeof(T) == typeof(ushort))
        {
            var ptr = MlxArray.DataUInt16(Array);
            
            return new MlxArrayEnumerator<ushort>(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }

        if (typeof(T) == typeof(uint))
        {
            var ptr = MlxArray.DataUInt32(Array);
            
            return new MlxArrayEnumerator<uint>(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }

        if (typeof(T) == typeof(ulong))
        {
            var ptr = MlxArray.DataUInt64(Array);
            
            return new MlxArrayEnumerator<ulong>(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }

        if (typeof(T) == typeof(sbyte))
        {
            var ptr = MlxArray.DataInt8(Array);
            
            return new MlxArrayEnumerator<sbyte>(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }

        if (typeof(T) == typeof(short))
        {
            var ptr = MlxArray.DataInt16(Array);
            
            return new MlxArrayEnumerator<short>(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }

        if (typeof(T) == typeof(int))
        {
            var ptr = MlxArray.DataInt32(Array);
            
            return new MlxArrayEnumerator<int>(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }

        if (typeof(T) == typeof(long))
        {
            var ptr = MlxArray.DataInt64(Array);
            
            return new MlxArrayEnumerator<long>(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }

        if (typeof(T) == typeof(Half))
        {
            var ptr = (Half*)MlxArray.DataFloat16(Array);
            
            return new MlxArrayEnumerator<Half>(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }

        if (typeof(T) == typeof(float))
        {
            var ptr = MlxArray.DataFloat32(Array);
            
            return new MlxArrayEnumerator<float>(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }

        if (typeof(T) == typeof(double))
        {
            var ptr = MlxArray.DataFloat64(Array);
            
            return new MlxArrayEnumerator<double>(ptr, size) as IEnumerator<T> ?? throw new InvalidOperationException();
        }
        
        throw new TensorTypeNotSupportedException(typeof(T));
    }

    public void Eval()
    {
        MlxArray.Eval(Array);
    }
}