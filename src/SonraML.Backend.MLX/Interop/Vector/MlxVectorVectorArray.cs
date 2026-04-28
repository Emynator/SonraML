using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop.Vector;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxVectorVectorArray
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_vector_array_new")]
    public static partial MlxVectorVectorArray New();

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_vector_array_new_data")]
    public static partial MlxVectorVectorArray NewData(ref readonly MlxVectorArray data, UIntPtr size);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_vector_array_new_value")]
    public static partial MlxVectorVectorArray NewValue(MlxVectorArray val);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_vector_array_free")]
    public static partial int Free(MlxVectorVectorArray vec);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_vector_array_size")]
    public static partial UIntPtr Size(MlxVectorVectorArray vec);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_vector_array_get")]
    public static partial int Get(ref readonly MlxVectorArray res, MlxVectorVectorArray vec, UIntPtr idx);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_vector_array_set")]
    public static partial int Set(ref readonly MlxVectorVectorArray vec, MlxVectorVectorArray src);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_vector_array_set_data")]
    public static partial int SetData
        (
        ref readonly MlxVectorVectorArray vec,
        ref readonly MlxVectorArray data,
        UIntPtr size
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_vector_array_set_value")]
    public static partial int SetValue(ref readonly MlxVectorVectorArray vec, MlxVectorArray val);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_vector_array_append_data")]
    public static partial int AppendData(MlxVectorVectorArray vec, ref readonly MlxVectorArray data, UIntPtr size);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_vector_array_append_value")]
    public static partial int AppendValue(MlxVectorVectorArray vec, MlxVectorArray val);
}