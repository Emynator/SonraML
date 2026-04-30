using SonraML.Backend.MLX.Managed;

namespace SonraML.Backend.MLX.Interfaces;

internal interface IMlxBackendGlobals
{
    public ManagedMlxStream? Stream { get; set; }
}