using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace SonraML.Core.Services;

public abstract class Worker : BackgroundService
{
    private readonly IServiceProvider serviceProvider;

    public Worker(IServiceProvider serviceProvider)
    {
        this.serviceProvider = serviceProvider;
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        var scope = serviceProvider.CreateAsyncScope();
        
        await scope.DisposeAsync();
    }
}