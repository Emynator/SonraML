using System.Runtime.InteropServices;

namespace SonraML.Backend.MLX.Interop.Io;

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct MlxIoVtable
{
    public delegate*<void*, byte> IsOpen;

    public delegate*<void*, byte> Good;

    public delegate*<void*, UIntPtr> Tell;

    public delegate*<void*, UIntPtr, int, void> Seek;

    public delegate*<void*, byte*, UIntPtr, void> Read;

    public delegate*<void*, byte*, UIntPtr, UIntPtr, void> ReadAtOffset;

    public delegate*<void*, byte*, UIntPtr, void> Write;

    public delegate*<void*, IntPtr> Label;

    public delegate*<void*, void> Free;
}