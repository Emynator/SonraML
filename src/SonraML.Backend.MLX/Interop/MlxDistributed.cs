using System.Runtime.InteropServices;
using SonraML.Backend.MLX.Interop.Enums;

namespace SonraML.Backend.MLX.Interop;

internal static unsafe partial class MlxDistributed
{
    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_is_available", StringMarshalling = StringMarshalling.Utf8)]
    [return: MarshalAs(UnmanagedType.U1)]
    public static partial bool IsAvailable(string bk);

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_all_gather")]
    public static partial int AllGather
        (
        ref readonly MlxArray res,
        MlxArray x,
        MlxDistributedGroup group,
        MlxStream S
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_all_max")]
    public static partial int AllMax
        (
        ref readonly MlxArray res,
        MlxArray x,
        MlxDistributedGroup group,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_all_min")]
    public static partial int AllMin
        (
        ref readonly MlxArray res,
        MlxArray x,
        MlxDistributedGroup group,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_all_sum")]
    public static partial int AllSum
        (
        ref readonly MlxArray res,
        MlxArray x,
        MlxDistributedGroup group,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_recv")]
    public static partial int Recv
        (
        ref readonly MlxArray res,
        int* shape,
        UIntPtr shapeNum,
        DType dtype,
        int src,
        MlxDistributedGroup group,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_recv_like")]
    public static partial int RecvLike
        (
        ref readonly MlxArray res,
        MlxArray x,
        int src,
        MlxDistributedGroup group,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_send")]
    public static partial int Send
        (
        ref readonly MlxArray res,
        MlxArray x,
        int dst,
        MlxDistributedGroup group,
        MlxStream s
        );

    [LibraryImport("mlxc", EntryPoint = "mlx_distributed_sum_scatter")]
    public static partial int SumScatter
        (
        ref readonly MlxArray res,
        MlxArray x,
        MlxDistributedGroup group,
        MlxStream s
        );
}