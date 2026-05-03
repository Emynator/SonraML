using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.ExecutionManagement;

internal abstract record class MlxResponse();

internal record class ErrorResponse(string Message) : MlxResponse; 

internal record class SuccessResponse() : MlxResponse;

internal record class TensorResponse(Guid Id, DType Type) : SuccessResponse;

internal record class TensorArrayResponse(List<Guid> Id, DType Type) : SuccessResponse;

internal record class EmptyTensorArrayResponse() : SuccessResponse;

internal record class EnumeratorResponse(object Enumerator) : SuccessResponse;

internal record class IsScalarResponse(bool IsScalar) : SuccessResponse;

internal record class ShapeResponse(TensorShape Shape) : SuccessResponse;

internal record class EqualsResponse(bool Value) : SuccessResponse;

internal record class ToStringResponse(string Value) : SuccessResponse;