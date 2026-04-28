using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop.Vector;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxVectorString
{
    private void* ctx;
    
    [LibraryImport("mlxc", EntryPoint = "mlx_vector_string_new")]
    public static partial MlxVectorString New();

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_string_new_data", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxVectorString NewData(string[] data, UIntPtr size);
        
    [LibraryImport("mlxc", EntryPoint = "mlx_vector_string_new_value", StringMarshalling = StringMarshalling.Utf8)]
    public static partial MlxVectorString NewValue(string val);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_string_free")]
    public static partial int Free(MlxVectorString vec);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_string_size")]
    public static partial UIntPtr Size(MlxVectorString vec);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_string_get")]
    public static partial int Get(IntPtr* res, MlxVectorString vec, UIntPtr idx);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_string_set")]
    public static partial int Set(MlxVectorString* vec, MlxVectorString src);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_string_set_data", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SetData(MlxVectorString* vec, string[] data, UIntPtr size);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_string_set_value", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int SetValue(MlxVectorString* vec, string val);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_string_append_data", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int AppendData(MlxVectorString vec, string[] data, UIntPtr size);

    [LibraryImport("mlxc", EntryPoint = "mlx_vector_string_append_value", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int AppendValue(MlxVectorString vec, string val);
}