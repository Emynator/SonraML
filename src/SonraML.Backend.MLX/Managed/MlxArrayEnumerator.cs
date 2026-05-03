using System.Collections;

namespace SonraML.Backend.MLX.Managed;

internal unsafe class MlxArrayEnumerator<T> : IEnumerator<T> where T : struct
{
    private readonly T* nativePtr;
    private readonly UIntPtr size;
    private UIntPtr currentIndex;

    public MlxArrayEnumerator(T* nativePtr, UIntPtr size)
    {
        this.nativePtr = nativePtr;
        this.size = size;
    }
    
    public bool MoveNext()
    {
        currentIndex++;
        if (currentIndex >= size)
        {
            return false;
        }
        
        return true;
    }

    public void Reset()
    {
        currentIndex = 0;
    }

    T IEnumerator<T>.Current => nativePtr[currentIndex];

    object? IEnumerator.Current => nativePtr[currentIndex];

    public void Dispose()
    {
    }
}