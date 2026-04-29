namespace SonraML.Backend.MLX.Managed;

public unsafe class MlxArrayBoolEnumerator : IEnumerator<bool>
{
    private readonly byte* nativePtr;
    private readonly UIntPtr size;
    private UIntPtr currentIndex;

    public MlxArrayBoolEnumerator(byte* nativePtr, UIntPtr size)
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

    bool IEnumerator<bool>.Current => nativePtr[currentIndex] != 0;

    public object? Current => nativePtr[currentIndex] != 0;

    public void Dispose()
    {
    }
}