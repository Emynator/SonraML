using System.ComponentModel.DataAnnotations;

namespace SonraML.Core.Config;

public class SonraRunnerConfigurations
{
    public List<SonraRunnerConfiguration> Configurations { get; set; } = [];
}

public class SonraRunnerConfiguration
{
    [Required]
    public string RunnerName { get; set; }

    public int Epochs { get; set; } = 0;
}