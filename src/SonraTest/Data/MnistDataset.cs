using System.Buffers.Binary;
using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;

namespace SonraTest.Data;

public class MnistDataset : IDisposable
{
    private readonly FileStream labelFile;
    private readonly BinaryReader labelReader;
    private readonly FileStream imageFile;
    private readonly BinaryReader imageReader;
    private readonly int count;
    private int current = 0;

    public MnistDataset(string imagePath, string labelPath)
    {
        if (!File.Exists(labelPath))
        {
            throw new FileNotFoundException($"File not found: '{labelPath}'.");
        }

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"File not found: '{imagePath}'.");
        }

        labelFile = File.OpenRead(labelPath);
        labelReader = new BinaryReader(labelFile);

        var buffer = labelReader.ReadBytes(4);
        var magic = BinaryPrimitives.ReadInt32BigEndian(buffer);
        if (magic != 2049)
        {
            throw new InvalidDataException($"Invalid MNIST label magic number: {magic}.");
        }

        buffer = labelReader.ReadBytes(4);
        var labelCount = BinaryPrimitives.ReadInt32BigEndian(buffer);

        imageFile = File.OpenRead(imagePath);
        imageReader = new BinaryReader(imageFile);

        buffer = imageReader.ReadBytes(4);
        magic = BinaryPrimitives.ReadInt32BigEndian(buffer);
        if (magic != 2051)
        {
            throw new InvalidDataException($"Invalid MNIST image magic number: {magic}.");
        }

        buffer = imageReader.ReadBytes(4);
        var imageCount = BinaryPrimitives.ReadInt32BigEndian(buffer);
        buffer = imageReader.ReadBytes(4);
        var rows = BinaryPrimitives.ReadInt32BigEndian(buffer);
        buffer = imageReader.ReadBytes(4);
        var columns = BinaryPrimitives.ReadInt32BigEndian(buffer);

        if (imageCount != labelCount)
        {
            throw new InvalidDataException($"Image count is {imageCount}, but label count is {labelCount}.");
        }

        count = imageCount;

        ImageSize = rows * columns;
    }

    public int ImageSize { get; }

    public MnistData? GetNext()
    {
        if (current >= count)
        {
            return null;
        }
        
        current++;
        var label = labelReader.ReadByte();
        var image = imageReader.ReadBytes(ImageSize);

        return new MnistData(image, label);
    }

    public void Dispose()
    {
        labelReader.Dispose();
        labelFile.Dispose();

        imageReader.Dispose();
        imageFile.Dispose();
    }
}