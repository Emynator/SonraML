using SonraML.Backend.MLX.ExecutionManagement;
using SonraML.Core.Enums;
using SonraML.Core.Interfaces;

namespace SonraML.Backend.MLX.Implementations;

internal class MlxBackend : ISonraMLBackend
{
    private readonly MlxBackendGlobals globals;
    private readonly MlxScheduler scheduler;
    private readonly Thread schedulerThread;
    private readonly CancellationTokenSource cts;

    public MlxBackend(MlxBackendGlobals globals, MlxScheduler scheduler)
    {
        this.globals = globals;
        this.scheduler = scheduler;
        cts = new CancellationTokenSource();
        schedulerThread = new Thread(() => scheduler.Execute(cts.Token));
    }
    
    public void Init(BackendDeviceType type)
    {
        globals.DeviceType = type;
        schedulerThread.Start();
    }

    public void Dispose()
    {
        cts.Cancel();
    }
}