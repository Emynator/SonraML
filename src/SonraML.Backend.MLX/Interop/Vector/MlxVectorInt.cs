using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop.Vector;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxVectorInt
{
    private void* ctx;
    
    [LibraryImport("mlxc", EntryPoint = "mlx_vector_int_new")]
    public static partial MlxVectorInt New();

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_int_new_data")]
    public static partial MlxVectorInt NewData(int* data, UIntPtr size);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_vector_int_new_value")]
    public static partial MlxVectorInt NewValue(int val);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_int_free")]
    public static partial int Free(MlxVectorInt vec);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_int_size")]
    public static partial UIntPtr Size(MlxVectorInt vec);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_int_get")]
    public static partial int Get(int* res, MlxVectorInt vec, UIntPtr idx);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_int_append_value")]
    public static partial int Value(MlxVectorInt vec, int val);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_int_set")]
    public static partial int Set(MlxVectorInt* vec, MlxVectorInt src);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_int_set_data")]
    public static partial int SetData(MlxVectorInt* vec, int* data, UIntPtr size);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_int_set_value")]
    public static partial int SetValue(MlxVectorInt* vec, int val);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_int_append_data")]
    public static partial int AppendData(MlxVectorInt vec, int* data, UIntPtr size);
}