using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop;

[StructLayout(LayoutKind.Sequential)]
internal unsafe partial struct MlxDistributedGroup
{
    private void* ctx;

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_group_new")]
    public static partial MlxDistributedGroup New();

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_group_free")]
    public static partial int Free(MlxDistributedGroup group);

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_init", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Init
        (
        ref readonly MlxDistributedGroup res,
        [MarshalAs(UnmanagedType.U1)] bool strict,
        string? bk
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_group_size")]
    public static partial int Size(MlxDistributedGroup group);

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_group_rank")]
    public static partial int Rank(MlxDistributedGroup group);

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_group_split")]
    public static partial int Split
        (
        ref readonly MlxDistributedGroup res,
        MlxDistributedGroup group,
        int color,
        int key
        );
}