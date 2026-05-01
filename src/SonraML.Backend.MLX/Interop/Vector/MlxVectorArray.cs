using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop.Vector;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxVectorArray
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_array_new")]
    public static partial MlxVectorArray New();

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_array_new_data")]
    public static partial MlxVectorArray NewData(MlxArray* data, UIntPtr size);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_array_new_value")]
    public static partial MlxVectorArray NewValue(MlxArray val);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_array_free")]
    public static partial int Free(MlxVectorArray vec);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_array_size")]
    public static partial UIntPtr Size(MlxVectorArray vec);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_array_get")]
    public static partial int Get(ref readonly MlxArray res, MlxVectorArray vec, UIntPtr idx);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_array_set")]
    public static partial int Set(ref readonly MlxVectorArray vec, MlxVectorArray src);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_array_set_data")]
    public static partial int SetData(ref readonly MlxVectorArray vec, ref readonly MlxArray data, UIntPtr size);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_array_set_value")]
    public static partial int SetValue(ref readonly MlxVectorArray vec, MlxArray val);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_array_append_data")]
    public static partial int AppendData(MlxVectorArray vec, ref readonly MlxArray data, UIntPtr size);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_array_append_value")]
    public static partial int AppendValue(MlxVectorArray vec, MlxArray val);
}