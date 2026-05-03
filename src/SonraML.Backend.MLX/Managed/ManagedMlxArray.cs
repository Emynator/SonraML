using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Core.Exceptions;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.Managed;

internal unsafe class ManagedMlxArray : IDisposable
{
    public MlxArray Array;

    public ManagedMlxArray()
    {
        Array = MlxArray.New();
    }

    public ManagedMlxArray(Memory<bool> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Bool);
    }

    public ManagedMlxArray(Memory<byte> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.UInt8);
    }

    public ManagedMlxArray(Memory<ushort> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.UInt16);
    }

    public ManagedMlxArray(Memory<uint> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.UInt32);
    }

    public ManagedMlxArray(Memory<ulong> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.UInt64);
    }

    public ManagedMlxArray(Memory<sbyte> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Int8);
    }

    public ManagedMlxArray(Memory<short> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Int16);
    }

    public ManagedMlxArray(Memory<int> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Int32);
    }

    public ManagedMlxArray(Memory<long> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Int64);
    }

    public ManagedMlxArray(Memory<Half> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Float16);
    }

    public ManagedMlxArray(Memory<float> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Float32);
    }

    public ManagedMlxArray(Memory<double> array, TensorShape shape)
    {
        using var arrayHandle = array.Pin();
        using var shapeHandle = shape.Shape.AsMemory().Pin();
        var shapePtr = (int*)shapeHandle.Pointer;
        Array = MlxArray.NewData(arrayHandle.Pointer, shapePtr, shape.Dimensions, DType.Float64);
    }

    public ManagedMlxArray(bool scalar)
    {
        Array = MlxArray.NewBool(scalar);
    }

    public ManagedMlxArray(byte scalar)
    {
        Array = MlxArray.NewData(&scalar, null, 0, DType.UInt8);
    }

    public ManagedMlxArray(ushort scalar)
    {
        Array = MlxArray.NewData(&scalar, null, 0, DType.UInt16);
    }

    public ManagedMlxArray(uint scalar)
    {
        Array = MlxArray.NewData(&scalar, null, 0, DType.UInt32);
    }

    public ManagedMlxArray(ulong scalar)
    {
        Array = MlxArray.NewData(&scalar, null, 0, DType.UInt64);
    }

    public ManagedMlxArray(sbyte scalar)
    {
        Array = MlxArray.NewData(&scalar, null, 0, DType.Int8);
    }

    public ManagedMlxArray(short scalar)
    {
        Array = MlxArray.NewData(&scalar, null, 0, DType.Int16);
    }

    public ManagedMlxArray(int scalar)
    {
        Array = MlxArray.NewInt(scalar);
    }

    public ManagedMlxArray(long scalar)
    {
        Array = MlxArray.NewData(&scalar, null, 0, DType.Int64);
    }

    public ManagedMlxArray(Half scalar)
    {
        Array = MlxArray.NewData(&scalar, null, 0, DType.Float16);
    }

    public ManagedMlxArray(float scalar)
    {
        Array = MlxArray.NewFloat32(scalar);
    }

    public ManagedMlxArray(double scalar)
    {
        Array = MlxArray.NewFloat64(scalar);
    }

    public DType Type => MlxArray.DType(Array);

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

    public void CopyFrom(ManagedMlxArray array)
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

    public bool GetBoolScalar()
    {
        Eval();
        MlxArray.ItemBool(out var res, Array);
        
        return res;
    }

    public byte GetByteScalar()
    {
        Eval();
        MlxArray.ItemUInt8(out var res, Array);
        
        return res;
    }

    public ushort GetUShortScalar()
    {
        Eval();
        MlxArray.ItemUInt16(out var res, Array);
        
        return res;
    }

    public uint GetUIntScalar()
    {
        Eval();
        MlxArray.ItemUInt32(out var res, Array);
        
        return res;
    }

    public ulong GetULongScalar()
    {
        Eval();
        MlxArray.ItemUInt64(out var res, Array);
        
        return res;
    }

    public sbyte GetSByteScalar()
    {
        Eval();
        MlxArray.ItemInt8(out var res, Array);
        
        return res;
    }

    public short GetShortScalar()
    {
        Eval();
        MlxArray.ItemInt16(out var res, Array);
        
        return res;
    }

    public int GetIntScalar()
    {
        Eval();
        MlxArray.ItemInt32(out var res, Array);
        
        return res;
    }

    public long GetLongScalar()
    {
        Eval();
        MlxArray.ItemInt64(out var res, Array);
        
        return res;
    }

    public Half GetHalfScalar()
    {
        Eval();
        MlxArray.ItemFloat16(out var res, Array);
        
        return (Half)res;
    }

    public float GetFloatScalar()
    {
        Eval();
        MlxArray.ItemFloat32(out var res, Array);
        
        return res;
    }

    public double GetDoubleScalar()
    {
        Eval();
        MlxArray.ItemFloat64(out var res, Array);
        
        return res;
    }

    public bool[] GetBoolData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new bool[size];
        var ptr = MlxArray.DataBool(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i] != 0;
        }

        return res;
    }

    public byte[] GetByteData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new byte[size];
        var ptr = MlxArray.DataUInt8(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i];
        }

        return res;
    }

    public ushort[] GetUShortData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new ushort[size];
        var ptr = MlxArray.DataUInt16(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i];
        }

        return res;
    }

    public uint[] GetUIntData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new uint[size];
        var ptr = MlxArray.DataUInt32(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i];
        }

        return res;
    }

    public ulong[] GetULongData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new ulong[size];
        var ptr = MlxArray.DataUInt64(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i];
        }

        return res;
    }

    public sbyte[] GetSByteData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new sbyte[size];
        var ptr = MlxArray.DataInt8(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i];
        }

        return res;
    }

    public short[] GetShortData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new short[size];
        var ptr = MlxArray.DataInt16(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i];
        }

        return res;
    }

    public int[] GetIntData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new int[size];
        var ptr = MlxArray.DataInt32(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i];
        }

        return res;
    }

    public long[] GetLongData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new long[size];
        var ptr = MlxArray.DataInt64(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i];
        }

        return res;
    }

    public Half[] GetHalfData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new Half[size];
        var ptr = (Half*)MlxArray.DataFloat16(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i];
        }

        return res;
    }

    public float[] GetFloatData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new float[size];
        var ptr = MlxArray.DataFloat32(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i];
        }

        return res;
    }

    public double[] GetDoubleData()
    {
        Eval();
        var size = MlxArray.Size(Array);
        var res = new double[size];
        var ptr = MlxArray.DataFloat64(Array);
        for (UIntPtr i = 0; i < size; i++)
        {
            res[i] = ptr[i];
        }

        return res;
    }

    public IEnumerator<T> GetEnumerator<T>() where T : struct
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