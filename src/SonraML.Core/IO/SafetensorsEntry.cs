using System.Text.Json.Serialization;
using SonraML.Core.Types;

namespace SonraML.Core.IO;

public abstract record class SafetensorsEntry();

public sealed record class SafetensorsDataEntry(GenericTensor Data)  : SafetensorsEntry;

public sealed record class SafetensorsHeaderEntry
    (
    [property: JsonPropertyName("dtype")] string DataType,
    [property: JsonPropertyName("shape")] int[] Shape,
    [property: JsonPropertyName("data_offsets")]
    long[] DataOffsets
    ) : SafetensorsEntry;