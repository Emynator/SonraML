using SonraML.Backend.MLX.Implementations;
using SonraML.Backend.MLX.Interop;
using SonraML.Backend.MLX.Interop.Vector;

namespace SonraML.Backend.MLX.Managed;

internal unsafe class ManagedMlxVectorArray<T> : IDisposable where T : struct
{
    public readonly MlxVectorArray Vector;

    public ManagedMlxVectorArray()
    {
        Vector = MlxVectorArray.New();
    }

    public ManagedMlxVectorArray(MlxTensor<T>[] array)
    {
        using var handle = array.Select(t => t.Array.Array).ToArray().AsMemory().Pin();
        Vector = MlxVectorArray.NewData((MlxArray*)handle.Pointer, (UIntPtr)array.Length);
    }

    public void Dispose()
    {
        MlxVectorArray.Free(Vector);
    }

    public UIntPtr Size => MlxVectorArray.Size(Vector);

    public ManagedMlxArray<T> Get(UIntPtr index)
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

    public ManagedMlxArray<T>[] ToArray()
    {
        var result = new ManagedMlxArray<T>[Size];
        for (UIntPtr i = 0; i < Size; i++)
        {
            result[i] = Get(i);
        }
        
        return result;
    }
}