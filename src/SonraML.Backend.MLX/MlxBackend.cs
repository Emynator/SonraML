using SonraML.Backend.MLX.Managed;
using SonraML.Core;
using SonraML.Core.Enums;
using SonraML.Core.Exceptions;

namespace SonraML.Backend.MLX;

internal class MlxBackend : SonraMLBackend
{
    private ManagedMlxStream? stream;

    public MlxBackend()
    {
        TensorFactory = new MlxTensorFactory();
    }

    internal static MlxBackend Instance =>
        SonraMLConfig.Backend as MlxBackend ?? throw new BackendNotInitializedException();

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