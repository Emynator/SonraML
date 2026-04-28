using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Enums;

namespace SonraML.Backend.MLX.Interop;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxArray
{
    private void* ctx;

    public static MlxArray Null()
    {
        return new() { ctx = null };
    }
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_tostring")]
    public static partial int ToString(out MlxString str, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_new")]
    public static partial MlxArray New();

    [LibraryImport("mlxc", EntryPoint = "mlx_array_free")]
    public static partial int Free(MlxArray arr);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_new_bool")]
    public static partial MlxArray NewBool([MarshalAs(UnmanagedType.U1)] bool val);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_new_int")]
    public static partial MlxArray NewInt(int val);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_new_float32")]
    public static partial MlxArray NewFloat32(float val);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_new_float64")]
    public static partial MlxArray NewFloat64(double val);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_new_complex")]
    public static partial MlxArray NewComplex(double realVal, double imagVal);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_new_data")]
    public static partial MlxArray NewData(void* data, int* shape, int dim, DType dType);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_set")]
    public static partial int Set(ref readonly MlxArray arr, MlxArray src);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_set_bool")]
    public static partial int SetBool(ref readonly MlxArray arr, [MarshalAs(UnmanagedType.U1)] bool val);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_set_int")]
    public static partial int SetInt(ref readonly MlxArray arr, int val);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_set_float32")]
    public static partial int SetFloat32(ref readonly MlxArray arr, float val);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_set_float64")]
    public static partial int SetFloat64(ref readonly MlxArray arr, double val);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_set_complex")]
    public static partial int SetComplex(ref readonly MlxArray arr, double realVal, double imagVal);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_set_data")]
    public static partial int SetData(ref readonly MlxArray arr, void* data, int* shape, int dim, DType dType);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_itemsize")]
    public static partial UIntPtr Itemsize(MlxArray arr);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_size")]
    public static partial UIntPtr Size(MlxArray arr);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_nbytes")]
    public static partial UIntPtr NBytes(MlxArray arr);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_ndim")]
    public static partial UIntPtr NDim(MlxArray arr);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_shape")]
    public static partial int* Shape(MlxArray arr);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_strides")]
    public static partial UIntPtr* Strides(MlxArray arr);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_dim")]
    public static partial int Dim(MlxArray arr);

    [LibraryImport("mlxc", EntryPoint = "mlx_array_dtype")]
    public static partial DType DType(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_eval")]
    public static partial int Eval(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_bool")]
    public static partial int ItemBool([MarshalAs(UnmanagedType.U1)] out bool res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_uint8")]
    public static partial int ItemUInt8(out byte res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_uint16")]
    public static partial int ItemUInt16(out ushort res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_uint32")]
    public static partial int ItemUInt32(out uint res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_uint64")]
    public static partial int ItemUInt64(out ulong res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_int8")]
    public static partial int ItemInt8(out sbyte res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_int16")]
    public static partial int ItemInt16(out short res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_int32")]
    public static partial int ItemInt32(out int res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_int64")]
    public static partial int ItemInt64(out long res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_float16")]
    public static partial int ItemFloat16(out ushort res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_float32")]
    public static partial int ItemFloat32(out float res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_item_float64")]
    public static partial int ItemFloat64(out double res, MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_bool")]
    public static partial byte* DataBool(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_uint8")]
    public static partial byte* DataUInt8(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_uint16")]
    public static partial ushort* DataUInt16(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_uint32")]
    public static partial uint* DataUInt32(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_uint64")]
    public static partial ulong* DataUInt64(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_int8")]
    public static partial sbyte* DataInt8(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_int16")]
    public static partial short* DataInt16(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_int32")]
    public static partial int* DataInt32(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_int64")]
    public static partial long* DataInt64(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_float16")]
    public static partial ushort* DataFloat16(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_float32")]
    public static partial float* DataFloat32(MlxArray arr);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_array_data_float64")]
    public static partial double* DataFloat64(MlxArray arr);
}