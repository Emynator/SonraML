using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using SonraML.Core.Config;
using SonraML.Core.Exceptions;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.Services;

internal class SonraWorker : BackgroundService
{
    private readonly ILogger<SonraWorker> logger;
    private readonly IServiceProvider serviceProvider;
    private readonly string runnerName;
    private readonly string contextName;

    public SonraWorker
        (
        IServiceProvider serviceProvider,
        string runnerName,
        string contextName
        )
    {
        logger = serviceProvider.GetRequiredService<ILogger<SonraWorker>>();
        this.serviceProvider = serviceProvider;
        this.runnerName = runnerName;
        this.contextName = contextName;
    }

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        logger.LogInformation("{time} - Initializing Runner '{name}'...", DateTime.Now, runnerName);
        var contextScope = serviceProvider.CreateAsyncScope();

        var context = contextScope.ServiceProvider.GetKeyedService<ISonraRunnerContext>(contextName);
        if (context is null)
        {
            throw new RunnerNotFoundException(contextName);
        }

        var contextRunner = contextScope.ServiceProvider.GetKeyedService<SonraRunner>(runnerName)
            ?? throw new RunnerNotFoundException(runnerName);
        contextRunner.Context = context;
        await contextRunner.BeforeFirstRun(ct);
        contextRunner.Dispose();

        var configs = contextScope.ServiceProvider
            .GetRequiredService<IOptions<SonraRunnerConfigurations>>()
            .Value;
        var config = configs.Configurations
            .FirstOrDefault(c => c.RunnerName == runnerName);
        var maxRuns = config?.Epochs ?? -1;

        logger.LogInformation("{time} - Runner '{name}' initialized. Executing...", DateTime.Now, runnerName);

        var epoch = 1;
        while (!ct.IsCancellationRequested)
        {
            var timestamp = Stopwatch.GetTimestamp();
            logger.LogInformation
            (
                "{time} - Runner '{name}' - Executing epoch {epoch}...",
                DateTime.Now,
                runnerName,
                epoch
            );

            var iterScope = contextScope.ServiceProvider.CreateAsyncScope();

            var runner = iterScope.ServiceProvider.GetKeyedService<SonraRunner>(runnerName)
                ?? throw new RunnerNotFoundException(runnerName);
            runner.Context = context;

            await runner.Run(ct);

            await iterScope.DisposeAsync();

            var elapsed = Stopwatch.GetElapsedTime(timestamp);
            logger.LogInformation
            (
                "{time} - Runner '{name}' - Epoch {epoch} executed in {duration}.",
                DateTime.Now,
                runnerName,
                epoch,
                elapsed
            );
            epoch++;

            if (maxRuns > 0 && epoch > maxRuns)
            {
                logger.LogInformation("{time} - Runner '{name}' - Max Epoch count reached.", DateTime.Now, runnerName);
                break;
            }
        }

        logger.LogInformation
        (
            "{time} - Runner '{name}' - Execution finished. Cleaning up...",
            DateTime.Now,
            runnerName
        );
        await contextScope.DisposeAsync();
    }
}