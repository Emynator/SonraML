using System.Text.Json.Serialization;

namespace SonraML.Core.Types;

public sealed record class SafetensorsTensor
    (
    [property: JsonPropertyName("dtype")] string DataType,
    [property: JsonPropertyName("shape")] int[] Shape,
    [property: JsonPropertyName("data_offsets")] long[] DataOffsets
    );