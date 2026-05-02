using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SonraML.Core.Exceptions;
using SonraML.Core.Interfaces;

namespace SonraML.Core.Services;

public class SonraWorker : BackgroundService
{
    private readonly IServiceProvider serviceProvider;
    private readonly string runnerName;
    private readonly string contextName;

    public SonraWorker(IServiceProvider serviceProvider, string runnerName, string contextName)
    {
        this.serviceProvider = serviceProvider;
        this.runnerName = runnerName;
        this.contextName = contextName;
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        using var contextScope = serviceProvider.CreateScope();
        
        var context = contextScope.ServiceProvider.GetKeyedService<ISonraRunnerContext>(contextName);
        if (context is null)
        {
            throw new RunnerNotFoundException(contextName);
        }

        while (!ct.IsCancellationRequested)
        {
            var iterScope = contextScope.ServiceProvider.CreateAsyncScope();
            
            var runner = iterScope.ServiceProvider.GetKeyedService<ISonraRunner>(runnerName)
                ?? throw new RunnerNotFoundException(runnerName);
            runner.Context = context;
            
            await runner.Run(ct);
            
            await iterScope.DisposeAsync();
        }
    }
}