using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop;

internal static unsafe partial class MlxMemory
{
    [LibraryImport("mlxc", EntryPoint = "mlx_clear_cache")]
    public static partial int ClearCache();
    
    [LibraryImport("mlxc", EntryPoint = "mlx_get_active_memory")]
    public static partial int GetActiveMemory(ref readonly UIntPtr res);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_get_cache_memory")]
    public static partial int GetCacheMemory(ref readonly UIntPtr res);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_get_memory_limit")]
    public static partial int GetMemoryLimit(ref readonly UIntPtr res);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_get_peak_memory")]
    public static partial int GetPeakMemory(ref readonly UIntPtr res);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_reset_peak_memory")]
    public static partial int ResetPeakMemory();
    
    [LibraryImport("mlxc", EntryPoint = "mlx_set_cache_limit")]
    public static partial int SetCacheLimit(ref readonly UIntPtr res, UIntPtr limit);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_set_memory_limit")]
    public static partial int SetMemoryLimit(ref readonly UIntPtr res, UIntPtr limit);
    
    [LibraryImport("mlxc", EntryPoint = "mlx_set_wired_limit")]
    public static partial int SetWiredLimit(ref readonly UIntPtr res, UIntPtr limit);
}