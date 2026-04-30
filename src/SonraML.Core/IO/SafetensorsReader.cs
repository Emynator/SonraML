using System.Text;
using System.Text.Json;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.IO;

public class SafetensorsReader : ITensorReader, IDisposable
{
    private readonly ITensorFactory tf;
    private readonly BinaryReader reader;
    private readonly Dictionary<string, SafetensorsTensor> header = new();
    private readonly long dataOffset;

    public SafetensorsReader(string path, ITensorFactory tf)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(path);
        }

        this.tf = tf;
        reader = new(File.OpenRead(path));
        var buffer = reader.ReadBytes(8);
        var headerSize = BitConverter.ToInt64(buffer);

        var headerBuffer = reader.ReadBytes((int)headerSize);
        var jsonHeader = Encoding.UTF8.GetString(headerBuffer);
        header = JsonSerializer.Deserialize<Dictionary<string, SafetensorsTensor>>(jsonHeader)
            ?? throw new IOException($"Failed to parse file header for file '{path}'.");
        header.Remove("__metadata__");

        dataOffset = 8 + headerSize;
    }

    public void Dispose()
    {
        reader.Dispose();
    }

    public ICollection<GenericTensor> ReadTensors()
    {
        var result = new List<GenericTensor>();
        foreach (var entry in header)
        {
            var t = ReadTensor(entry.Value, entry.Key);
            if (t is not null)
            {
                result.Add(t);
            }
        }
        
        return result;
    }

    public GenericTensor? ReadTensor(string name)
    {
        if (!header.TryGetValue(name, out var entry))
        {
            return null;
        }

        return ReadTensor(entry, name);
    }

    private GenericTensor? ReadTensor(SafetensorsTensor entry, string name)
    {
        var length = entry.DataOffsets[1] - entry.DataOffsets[0];
        var shape = new TensorShape(entry.Shape);
        if (!SafetensorsHelper.DtypeToSize.TryGetValue(entry.DataType, out var typeSize))
        {
            throw new InvalidOperationException($"Dtype '{entry.DataType}' is not a valid safetensors dtype.");
        }

        reader.BaseStream.Seek(dataOffset + entry.DataOffsets[0], SeekOrigin.Begin);
        if (entry.DataType == "I8")
        {
            var buffer = new sbyte[length];
            for (var i = 0; i < length; i++)
            {
                buffer[i] = reader.ReadSByte();
            }

            return tf.Create(buffer.AsMemory(), shape, name);
        }

        var data = reader.ReadBytes((int)length);

        switch (entry.DataType)
        {
            case "BOOL":
                var bBuffer = new bool[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    bBuffer[i] = BitConverter.ToBoolean(data, i * typeSize);
                }

                return tf.Create(bBuffer.AsMemory(), shape, name);

            case "U8":
                return tf.Create(data.AsMemory(), shape, name);

            case "U16":
                var u16Buffer = new ushort[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    u16Buffer[i] = BitConverter.ToUInt16(data, i * typeSize);
                }

                return tf.Create(u16Buffer.AsMemory(), shape, name);

            case "U32":
                var u32Buffer = new uint[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    u32Buffer[i] = BitConverter.ToUInt32(data, i * typeSize);
                }

                return tf.Create(u32Buffer.AsMemory(), shape, name);

            case "U64":
                var u64Buffer = new ulong[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    u64Buffer[i] = BitConverter.ToUInt64(data, i * typeSize);
                }

                return tf.Create(u64Buffer.AsMemory(), shape, name);

            case "I16":
                var i16Buffer = new short[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    i16Buffer[i] = BitConverter.ToInt16(data, i * typeSize);
                }

                return tf.Create(i16Buffer.AsMemory(), shape, name);

            case "I32":
                var i32Buffer = new int[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    i32Buffer[i] = BitConverter.ToInt32(data, i * typeSize);
                }

                return tf.Create(i32Buffer.AsMemory(), shape, name);

            case "I64":
                var i64Buffer = new long[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    i64Buffer[i] = BitConverter.ToInt64(data, i * typeSize);
                }

                return tf.Create(i64Buffer.AsMemory(), shape, name);

            case "F16":
                var f16Buffer = new Half[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    f16Buffer[i] = BitConverter.ToHalf(data, i * typeSize);
                }

                return tf.Create(f16Buffer.AsMemory(), shape, name);

            case "F32":
                var f32Buffer = new float[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    f32Buffer[i] = BitConverter.ToSingle(data, i * typeSize);
                }

                return tf.Create(f32Buffer.AsMemory(), shape, name);

            case "F64":
                var f64Buffer = new double[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    f64Buffer[i] = BitConverter.ToDouble(data, i * typeSize);
                }

                return tf.Create(f64Buffer.AsMemory(), shape, name);
        }

        return null;
    }
}