using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Managed;
using SonraML.Core;
using SonraML.Core.Enums;
using SonraML.Core.Exceptions;

namespace SonraML.Backend.MLX;

internal unsafe class MlxBackend : SonraMLBackend
{
    private ManagedMlxStream? stream;

    public MlxBackend()
    {
        TensorFactory = new MlxTensorFactory();
        // Mlx.SetErrorHandler(null, null, null);
    }

    internal static MlxBackend Instance =>
        SonraMLConfiguration.Backend as MlxBackend ?? throw new BackendNotInitializedException();

    internal ManagedMlxStream Stream => stream ?? throw new BackendNotInitializedException();

    public override void Init(BackendDeviceType type)
    {
        stream = new(type);
    }

    public override void Dispose()
    {
        Stream.Dispose();
    }
}