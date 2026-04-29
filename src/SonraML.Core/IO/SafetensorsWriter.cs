using System.Text;
using System.Text.Json;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.IO;

public class SafetensorsWriter : ITensorWriter, IDisposable
{
    private readonly BinaryWriter writer;
    
    public SafetensorsWriter(string path, bool overwrite = false)
    {
        var dir = Path.GetDirectoryName(path);
        if (!Directory.Exists(dir) && dir is not null)
        {
            Directory.CreateDirectory(dir);
        }

        if (File.Exists(path))
        {
            if (overwrite)
            {
                File.Delete(path);
            }
            else
            {
                throw new IOException($"File '{path}' already exists.");
            }
        }
        
        writer = new(File.OpenWrite(path));
    }
    
    public void Dispose()
    {
        writer.Dispose();
    }

    public void WriteTensor(GenericTensor tensor)
    {
        throw new NotSupportedException();
    }

    public void WriteTensors(ICollection<GenericTensor> tensors)
    {
        var header = new Dictionary<string, SafetensorsTensor>();
        var offset = 0;
        foreach (var tensor in tensors)
        {
            if (header.ContainsKey(tensor.Name))
            {
                throw new ArgumentException($"Duplicate name '{tensor.Name}'.");
            }

            if (!SafetensorsHelper.TypeToDtype.TryGetValue(tensor.Type, out var dtype))
            {
                throw new ArgumentException($"Safetensors don't support type '{tensor.Type.Name}'.");
            }

            if (!SafetensorsHelper.DtypeToSize.TryGetValue(dtype, out var typeSize))
            {
                throw new ArgumentException($"Safetensors don't support dtype '{dtype}'.");
            }
            
            var size = typeSize * tensor.Shape.Size;
            header[tensor.Name] = new(dtype, tensor.Shape.Shape, [offset, offset + size]);
        }
        
        var jsonString = JsonSerializer.Serialize(header);
        var headerBytes = Encoding.UTF8.GetBytes(jsonString);
        var headerSize = (long)headerBytes.Length;
        
        writer.Write(headerSize);
        writer.Write(headerBytes);

        foreach (var tensor in tensors)
        {
            if (tensor.Type == typeof(bool))
            {
                var t = tensor.AsTensor<bool>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    byte val = (byte)(enumerator.Current ? 1 : 0);
                    writer.Write(val);
                }

                continue;
            }

            if (tensor.Type == typeof(byte))
            {
                var t = tensor.AsTensor<byte>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }
            
            if (tensor.Type == typeof(ushort))
            {
                var t = tensor.AsTensor<ushort>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }
            
            if (tensor.Type == typeof(uint))
            {
                var t = tensor.AsTensor<uint>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }
            
            if (tensor.Type == typeof(ulong))
            {
                var t = tensor.AsTensor<ulong>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }
            
            if (tensor.Type == typeof(sbyte))
            {
                var t = tensor.AsTensor<sbyte>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }
            
            if (tensor.Type == typeof(short))
            {
                var t = tensor.AsTensor<byte>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }
            
            if (tensor.Type == typeof(int))
            {
                var t = tensor.AsTensor<int>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }
            
            if (tensor.Type == typeof(long))
            {
                var t = tensor.AsTensor<long>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }
            
            if (tensor.Type == typeof(Half))
            {
                var t = tensor.AsTensor<Half>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }
            
            if (tensor.Type == typeof(float))
            {
                var t = tensor.AsTensor<float>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }
            
            if (tensor.Type == typeof(double))
            {
                var t = tensor.AsTensor<double>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }
            
            throw new ArgumentException($"Safetensors don't support type '{tensor.Type.Name}'.");
        }
        
        writer.Flush();
    }
}