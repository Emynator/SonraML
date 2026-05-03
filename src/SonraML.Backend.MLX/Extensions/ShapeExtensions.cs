using System.Buffers;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.Extensions;

internal static class ShapeExtensions
{
    public static MemoryHandle GetHandle(this TensorShape shape)
    {
        return shape.Shape.AsMemory().Pin();
    }
}