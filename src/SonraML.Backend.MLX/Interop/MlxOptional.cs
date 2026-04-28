using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Enums;

namespace SonraML.Backend.MLX.Interop;

[StructLayout(LayoutKind.Sequential)]
internal struct MlxOptionalInt
{
    public static MlxOptionalInt None()
    {
        return new() { Value = 0, HasValue = 0 };
    }
    
    public MlxOptionalInt(int value)
    {
        Value = value;
        HasValue = 1;
    }
    
    public int Value;

    public byte HasValue;
}

[StructLayout(LayoutKind.Sequential)]
internal struct MlxOptionalFloat
{
    public static MlxOptionalFloat None()
    {
        return new() { Value = 0, HasValue = 0 };
    }
    
    public MlxOptionalFloat(float value)
    {
        Value = value;
        HasValue = 1;
    }
    
    public float Value;

    public byte HasValue;
}

[StructLayout(LayoutKind.Sequential)]
internal struct MlxOptionalDType
{
    public static MlxOptionalDType None()
    {
        return new() { Value = 0, HasValue = 0 };
    }
    
    public MlxOptionalDType(DType value)
    {
        Value = value;
        HasValue = 1;
    }
    
    public DType Value;

    public byte HasValue;
}