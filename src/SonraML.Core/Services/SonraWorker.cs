using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Core.Exceptions;
using SonraML.Core.Interfaces;

namespace SonraML.Core.Services;

public class SonraWorker : BackgroundService
{
    private readonly IServiceProvider serviceProvider;
    private readonly string runnerName;

    public SonraWorker(IServiceProvider serviceProvider, string runnerName)
    {
        this.serviceProvider = serviceProvider;
        this.runnerName = runnerName;
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            var scope = serviceProvider.CreateAsyncScope();
            
            var runner = scope.ServiceProvider.GetKeyedService<ISonraRunner>(runnerName)
                ?? throw new RunnerNotFoundException(runnerName);
            
            await runner.Run(ct);
            
            await scope.DisposeAsync();
        }
    }
}