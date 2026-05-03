using System.Runtime.InteropServices;
using SonraML.Core.Exceptions;

namespace SonraML.Backend.MLX.Interop.Enums;

internal enum DType
{
    Bool,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    BFloat16,
    Complex64,
}

internal static partial class MlxDType
{
    [LibraryImport("mlxc", EntryPoint = "mlx_dtype_size")]
    public static partial UIntPtr Size(DType dtype);

    public static DType? GetDType<T>()
    {
        if (typeof(T) == typeof(bool))
        {
            return DType.Bool;
        }
        
        if (typeof(T) == typeof(byte))
        {
            return DType.UInt8;
        }
        
        if (typeof(T) == typeof(ushort))
        {
            return DType.UInt16;
        }
        
        if (typeof(T) == typeof(uint))
        {
            return DType.UInt32;
        }
        
        if (typeof(T) == typeof(ulong))
        {
            return DType.UInt64;
        }
        
        if (typeof(T) == typeof(sbyte))
        {
            return DType.Int8;
        }
        
        if (typeof(T) == typeof(short))
        {
            return DType.Int16;
        }
        
        if (typeof(T) == typeof(int))
        {
            return DType.Int32;
        }
        
        if (typeof(T) == typeof(long))
        {
            return DType.Int64;
        }
        
        if (typeof(T) == typeof(Half))
        {
            return DType.Float16;
        }
        
        if (typeof(T) == typeof(float))
        {
            return DType.Float32;
        }
        
        if (typeof(T) == typeof(double))
        {
            return DType.Float64;
        }

        return null;
    }

    public static DType GetDTypeValid<T>()
    {
        if (typeof(T) == typeof(bool))
        {
            return DType.Bool;
        }
        
        if (typeof(T) == typeof(byte))
        {
            return DType.UInt8;
        }
        
        if (typeof(T) == typeof(ushort))
        {
            return DType.UInt16;
        }
        
        if (typeof(T) == typeof(uint))
        {
            return DType.UInt32;
        }
        
        if (typeof(T) == typeof(ulong))
        {
            return DType.UInt64;
        }
        
        if (typeof(T) == typeof(sbyte))
        {
            return DType.Int8;
        }
        
        if (typeof(T) == typeof(short))
        {
            return DType.Int16;
        }
        
        if (typeof(T) == typeof(int))
        {
            return DType.Int32;
        }
        
        if (typeof(T) == typeof(long))
        {
            return DType.Int64;
        }
        
        if (typeof(T) == typeof(Half))
        {
            return DType.Float16;
        }
        
        if (typeof(T) == typeof(float))
        {
            return DType.Float32;
        }
        
        if (typeof(T) == typeof(double))
        {
            return DType.Float64;
        }

        throw new BackendOperationException("DType was invalid.");
    }

    public static Type? GetType(DType dtype)
    {
        return dtype switch
        {
            DType.Bool => typeof(bool),
            DType.UInt8 => typeof(byte),
            DType.UInt16 => typeof(ushort),
            DType.UInt32 => typeof(uint),
            DType.UInt64 => typeof(ulong),
            DType.Int8 => typeof(sbyte),
            DType.Int16 => typeof(short),
            DType.Int32 => typeof(int),
            DType.Int64 => typeof(long),
            DType.Float16 => typeof(Half),
            DType.Float32 => typeof(float),
            DType.Float64 => typeof(double),
            _ => null,
        };
    }

    public static Type GetTypeValid(DType dtype)
    {
        return dtype switch
        {
            DType.Bool => typeof(bool),
            DType.UInt8 => typeof(byte),
            DType.UInt16 => typeof(ushort),
            DType.UInt32 => typeof(uint),
            DType.UInt64 => typeof(ulong),
            DType.Int8 => typeof(sbyte),
            DType.Int16 => typeof(short),
            DType.Int32 => typeof(int),
            DType.Int64 => typeof(long),
            DType.Float16 => typeof(Half),
            DType.Float32 => typeof(float),
            DType.Float64 => typeof(double),
            _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
        };
    }
}