using System.Text;
using System.Text.Json;
using SonraML.Core.Extensions;
using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.IO;

public class SafetensorsStore : ITensorStore, IAsyncDisposable
{
    private readonly ITensorFactory tf;
    private readonly string filePath;
    private readonly FileStream file;
    private readonly SemaphoreSlim rwLock = new(1, 1);
    private Dictionary<string, SafetensorsEntry>? header = null;
    private long dataOffset = 0;

    public SafetensorsStore(ITensorFactory tf, string filePath)
    {
        this.tf = tf;
        this.filePath = filePath;
        var exists = false;
        if (File.Exists(filePath))
        {
            exists = true;
            file = File.Open(filePath, FileMode.Open);
        }
        else if (Directory.Exists(Path.GetDirectoryName(filePath)))
        {
            file = File.Create(filePath);
        }
        else
        {
            throw new FileNotFoundException($"Cannot find file: {filePath}");
        }
    }

    private Dictionary<string, SafetensorsEntry> Header
    {
        get
        {
            if (header is null)
            {
                ReadHeader();
            }

            return header!;
        }
    }

    public async ValueTask DisposeAsync()
    {
        await Task.Run(() => SaveFile(true));
        await file.DisposeAsync();
        rwLock.Dispose();
    }

    public void Dispose()
    {
        DisposeAsync().AsTask().Wait();
    }

    public async Task<ICollection<string>> ListTensors()
    {
        await rwLock.WaitAsync();
        var result = Header.Select(t => t.Key).ToList();
        rwLock.Release();

        return result;
    }

    public async Task<bool> Contains(string key)
    {
        await rwLock.WaitAsync();
        var result = Header.ContainsKey(key);
        rwLock.Release();

        return result;
    }

    public async Task<bool> Contains(ICollection<string> keys)
    {
        await rwLock.WaitAsync();
        foreach (var key in keys)
        {
            if (!Header.ContainsKey(key))
            {
                rwLock.Release();
                return false;
            }
        }
        rwLock.Release();

        return true;
    }

    public async Task<ICollection<GenericTensor>> LoadTensors()
    {
        var tasks = Header.Select(entry => LoadTensor(entry.Key));
        var result = await Task.WhenAll(tasks);

        return result.Where(t => t is not null).Select(t => t!).ToList();
    }

    public async Task<ICollection<GenericTensor>> LoadTensors(ICollection<string> keys)
    {
        await rwLock.WaitAsync();
        var tasks = keys.Select(k => LoadTensor(k, false));
        var result = await Task.WhenAll(tasks);
        rwLock.Release();

        return result.Where(t => t is not null).Select(t => t!).ToList();
    }

    public Task<GenericTensor?> LoadTensor(string key)
    {
        return LoadTensor(key, true);
    }

    public Task AddTensor(GenericTensor tensor)
    {
        return AddTensor(tensor, true);
    }

    public async Task AddTensors(ICollection<GenericTensor> tensors)
    {
        await rwLock.WaitAsync();
        var tasks = tensors.Select(t => AddTensor(t, false)).ToList();
        await Task.WhenAll(tasks);
        rwLock.Release();
    }

    public async Task RemoveTensor(string key)
    {
        await rwLock.WaitAsync();
        Header.Remove(key);
        rwLock.Release();
    }

    public Task Persist()
    {
        return Task.Run(() => SaveFile(false));
    }

    public async Task<GenericTensor?> LoadTensor(string key, bool isSingle)
    {
        GenericTensor? result = null;

        if (isSingle)
        {
            await rwLock.WaitAsync();
        }

        if (Header.TryGetValue(key, out var entry))
        {
            if (entry is SafetensorsDataEntry dataEntry)
            {
                result = dataEntry.Data;
            }

            if (entry is SafetensorsHeaderEntry headerEntry)
            {
                result = await Task.Run(() => ReadTensor(headerEntry, key));
            }
        }

        if (isSingle)
        {
            rwLock.Release();
        }

        return result;
    }

    private async Task AddTensor(GenericTensor tensor, bool isSingle)
    {
        if (isSingle)
        {
            await rwLock.WaitAsync();
        }

        var value = new SafetensorsDataEntry(tensor);
        if (!Header.TryAdd(tensor.Name, value))
        {
            Header[tensor.Name] = value;
        }

        if (isSingle)
        {
            rwLock.Release();
        }
    }

    private void ReadHeader()
    {
        file.Seek(0, SeekOrigin.Begin);
        using var binaryReader = new BinaryReader(file, Encoding.UTF8, true);
        var buffer = binaryReader.ReadBytes(8);
        var headerSize = BitConverter.ToInt64(buffer);

        var headerBuffer = binaryReader.ReadBytes((int)headerSize);
        var jsonHeader = Encoding.UTF8.GetString(headerBuffer);
        var rawHeader = JsonSerializer.Deserialize<Dictionary<string, SafetensorsHeaderEntry>>(jsonHeader)
            ?? throw new IOException($"Failed to parse file header for file '{filePath}'.");
        rawHeader.Remove("__metadata__");

        header = new();
        foreach (var headerEntry in rawHeader)
        {
            header.Add(headerEntry.Key, headerEntry.Value);
        }

        dataOffset = 8 + headerSize;
    }

    private GenericTensor? ReadTensor(SafetensorsHeaderEntry entry, string name)
    {
        var shape = new TensorShape(entry.Shape);
        if (!SafetensorsHelper.DtypeToSize.TryGetValue(entry.DataType, out var typeSize))
        {
            throw new InvalidOperationException($"Dtype '{entry.DataType}' is not a valid safetensors dtype.");
        }

        var buffer = ReadFromFile(entry, name);
        if (buffer.Length != shape.Size * typeSize)
        {
            throw new InvalidOperationException("Shapesize and size on file do not match.");
        }

        switch (entry.DataType)
        {
            case "BOOL":
                var bBuffer = new bool[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    bBuffer[i] = BitConverter.ToBoolean(buffer, i * typeSize);
                }

                return tf.FromArray(bBuffer, shape, name);

            case "U8":
            case "I8":
                return tf.FromArray(buffer, shape, name);

            case "U16":
                var u16Buffer = new ushort[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    u16Buffer[i] = BitConverter.ToUInt16(buffer, i * typeSize);
                }

                return tf.FromArray(u16Buffer, shape, name);

            case "U32":
                var u32Buffer = new uint[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    u32Buffer[i] = BitConverter.ToUInt32(buffer, i * typeSize);
                }

                return tf.FromArray(u32Buffer, shape, name);

            case "U64":
                var u64Buffer = new ulong[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    u64Buffer[i] = BitConverter.ToUInt64(buffer, i * typeSize);
                }

                return tf.FromArray(u64Buffer, shape, name);

            case "I16":
                var i16Buffer = new short[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    i16Buffer[i] = BitConverter.ToInt16(buffer, i * typeSize);
                }

                return tf.FromArray(i16Buffer, shape, name);

            case "I32":
                var i32Buffer = new int[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    i32Buffer[i] = BitConverter.ToInt32(buffer, i * typeSize);
                }

                return tf.FromArray(i32Buffer, shape, name);

            case "I64":
                var i64Buffer = new long[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    i64Buffer[i] = BitConverter.ToInt64(buffer, i * typeSize);
                }

                return tf.FromArray(i64Buffer, shape, name);

            case "F16":
                var f16Buffer = new Half[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    f16Buffer[i] = BitConverter.ToHalf(buffer, i * typeSize);
                }

                return tf.FromArray(f16Buffer, shape, name);

            case "F32":
                var f32Buffer = new float[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    f32Buffer[i] = BitConverter.ToSingle(buffer, i * typeSize);
                }

                return tf.FromArray(f32Buffer, shape, name);

            case "F64":
                var f64Buffer = new double[shape.Size];
                for (var i = 0; i < shape.Size; i++)
                {
                    f64Buffer[i] = BitConverter.ToDouble(buffer, i * typeSize);
                }

                return tf.FromArray(f64Buffer, shape, name);
        }

        return null;
    }

    private byte[] ReadFromFile(SafetensorsHeaderEntry entry, string name)
    {
        var length = entry.DataOffsets[1] - entry.DataOffsets[0];
        using var reader = new BinaryReader(file, Encoding.UTF8, true);
        reader.BaseStream.Seek(dataOffset + entry.DataOffsets[0], SeekOrigin.Begin);

        return reader.ReadBytes((int)length);
    }

    private void SaveFile(bool isDisposing)
    {
        rwLock.Wait();
        var unsaved = Header
            .Where(entry => entry.Value is SafetensorsDataEntry)
            .ToList();
        if (unsaved.Count == 0)
        {
            rwLock.Release();
            return;
        }

        var existing = Header
            .Where(entry => entry.Value is SafetensorsHeaderEntry)
            .ToList();

        var offset = 0;
        var oldEntriesToWrite = new List<byte[]>();
        var newHeader = new Dictionary<string, SafetensorsEntry>();
        foreach (var entry in existing)
        {
            if (entry.Value is not SafetensorsHeaderEntry headerEntry)
            {
                continue;
            }

            var data = ReadFromFile(headerEntry, entry.Key);
            newHeader.Add
            (
                entry.Key,
                headerEntry with
                {
                    DataOffsets = [offset, offset + data.Length],
                }
            );

            offset += data.Length;
            oldEntriesToWrite.Add(data);
        }

        var newEntriesToWrite = new List<GenericTensor>();
        foreach (var entry in unsaved)
        {
            if (entry.Value is not SafetensorsDataEntry dataEntry)
            {
                continue;
            }

            var type = dataEntry.Data.Type;

            if (!SafetensorsHelper.TypeToDtype.TryGetValue(type, out var dtype))
            {
                rwLock.Release();
                throw new ArgumentException($"Safetensors don't support type '{type.Name}'.");
            }

            if (!SafetensorsHelper.DtypeToSize.TryGetValue(dtype, out var typeSize))
            {
                rwLock.Release();
                throw new ArgumentException($"Safetensors don't support dtype '{dtype}'.");
            }

            var size = typeSize * dataEntry.Data.Shape.Size;
            newHeader.Add
            (
                entry.Key,
                new SafetensorsHeaderEntry(dtype, dataEntry.Data.Shape.Shape, [offset, offset + size])
            );
            newEntriesToWrite.Add(dataEntry.Data);
        }

        var jsonString = JsonSerializer.Serialize(newHeader);
        var headerBytes = Encoding.UTF8.GetBytes(jsonString);
        var headerSize = (long)headerBytes.Length;

        using var writer = new BinaryWriter(file, Encoding.UTF8, true);
        writer.Write(headerSize);
        writer.Write(headerBytes);

        foreach (var entry in oldEntriesToWrite)
        {
            writer.Write(entry);
        }

        foreach (var entry in newEntriesToWrite)
        {
            var type = entry.Type;
            if (type == typeof(bool))
            {
                var t = entry.AsTensor<bool>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    byte val = (byte)(enumerator.Current ? 1 : 0);
                    writer.Write(val);
                }

                continue;
            }

            if (type == typeof(byte))
            {
                var t = entry.AsTensor<byte>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }

            if (type == typeof(ushort))
            {
                var t = entry.AsTensor<ushort>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }

            if (type == typeof(uint))
            {
                var t = entry.AsTensor<uint>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }

            if (type == typeof(ulong))
            {
                var t = entry.AsTensor<ulong>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }

            if (type == typeof(sbyte))
            {
                var t = entry.AsTensor<sbyte>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }

            if (type == typeof(short))
            {
                var t = entry.AsTensor<byte>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }

            if (type == typeof(int))
            {
                var t = entry.AsTensor<int>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }

            if (type == typeof(long))
            {
                var t = entry.AsTensor<long>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }

            if (type == typeof(Half))
            {
                var t = entry.AsTensor<Half>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }

            if (type == typeof(float))
            {
                var t = entry.AsTensor<float>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }

            if (type == typeof(double))
            {
                var t = entry.AsTensor<double>();
                using var enumerator = t.GetEnumerator();
                while (enumerator.MoveNext())
                {
                    writer.Write(enumerator.Current);
                }

                continue;
            }

            rwLock.Release();
            throw new ArgumentException($"Safetensors don't support type '{type.Name}'.");
        }

        writer.Flush();

        if (!isDisposing)
        {
            ReadHeader();
        }
        
        rwLock.Release();
    }
}