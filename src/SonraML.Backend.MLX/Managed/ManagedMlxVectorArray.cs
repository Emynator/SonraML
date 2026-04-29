using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Managed;

internal class ManagedMlxVectorArray : IDisposable
{
    public MlxVectorArray Vector = MlxVectorArray.New();

    public void Dispose()
    {
        MlxVectorArray.Free(Vector);
    }

    public ManagedMlxArray<T> Get<T>(UIntPtr index) where T : struct
    {
        var size = MlxVectorArray.Size(Vector);
        if (index >= size)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }
        
        var result = new ManagedMlxArray<T>();
        MlxVectorArray.Get(in result.Array, Vector, index);
        
        return result;
    }
}