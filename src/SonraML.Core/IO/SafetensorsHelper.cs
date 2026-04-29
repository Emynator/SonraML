namespace SonraML.Core.IO;

public class SafetensorsHelper
{
    public static readonly Dictionary<string, Type> DtypeToType = new()
    {
        ["BOOL"] = typeof(bool),
        ["U8"] = typeof(byte),
        ["U16"] = typeof(ushort),
        ["U32"] = typeof(uint),
        ["U64"] = typeof(ulong),
        ["I8"] = typeof(sbyte),
        ["I16"] = typeof(short),
        ["I32"] = typeof(int),
        ["I64"] = typeof(long),
        ["F16"] = typeof(Half),
        ["F32"] = typeof(float),
        ["F64"] = typeof(double),
    };

    public static readonly Dictionary<string, int> DtypeToSize = new()
    {
        ["BOOL"] = 1,
        ["U8"] = 1,
        ["U16"] = 2,
        ["U32"] = 4,
        ["U64"] = 8,
        ["I8"] = 1,
        ["I16"] = 2,
        ["I32"] = 4,
        ["I64"] = 8,
        ["F16"] = 2,
        ["F32"] = 4,
        ["F64"] = 8,
    };

    public static readonly Dictionary<Type, string> TypeToDtype = new()
    {
        [typeof(bool)] = "BOOL",
        [typeof(byte)] = "U8",
        [typeof(ushort)] = "U16",
        [typeof(uint)] = "U32",
        [typeof(ulong)] = "U64",
        [typeof(sbyte)] = "I8",
        [typeof(short)] = "I16",
        [typeof(int)] = "I32",
        [typeof(long)] = "I64",
        [typeof(Half)] = "F16",
        [typeof(float)] = "F32",
        [typeof(double)] = "F64",
    };
}