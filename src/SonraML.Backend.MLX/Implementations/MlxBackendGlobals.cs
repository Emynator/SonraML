using SonraML.Backend.MLX.Interfaces;
using SonraML.Backend.MLX.Managed;

namespace SonraML.Backend.MLX.Implementations;

internal class MlxBackendGlobals : IMlxBackendGlobals
{
    public ManagedMlxStream? Stream { get; set; }
}