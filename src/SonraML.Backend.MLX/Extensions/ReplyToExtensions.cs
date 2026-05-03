using SonraML.Backend.MLX.ExecutionManagement;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.Extensions;

internal static class ReplyToExtensions
{
    public static void SetErrorResponse(this ReplyTo replyTo, string message)
    {
        replyTo.ResultQueue.Enqueue(new ErrorResponse(message));
        replyTo.Signal.Set();
    }
    
    public static void SetSuccessResponse(this ReplyTo replyTo)
    {
        replyTo.ResultQueue.Enqueue(new SuccessResponse());
        replyTo.Signal.Set();
    }

    public static void SetTensorResponse(this ReplyTo replyTo, MlxTensor t)
    {
        replyTo.ResultQueue.Enqueue(new TensorResponse(t.Id, t.Type));
        replyTo.Signal.Set();
    }
    
    public static void SetTensorArrayResponse(this ReplyTo replyTo, List<MlxTensor> ts)
    {
        if (ts.Count == 0)
        {
            replyTo.ResultQueue.Enqueue(new EmptyTensorArrayResponse());
            replyTo.Signal.Set();
        }
        
        var ids = ts.Select(t => t.Id).ToList();
        var type = ts.First().Type;
        if (ts.Any(t => t.Type != type))
        {
            replyTo.SetErrorResponse("Not all Tensors share the same type!");
        }
        
        replyTo.ResultQueue.Enqueue(new TensorArrayResponse(ids, type));
        replyTo.Signal.Set();
    }

    public static void SetEnumeratorResponse(this ReplyTo replyTo, object value)
    {
        replyTo.ResultQueue.Enqueue(new EnumeratorResponse(value));
        replyTo.Signal.Set();
    }
    
    public static void SetIsScalarResponse(this ReplyTo replyTo, bool value)
    {
        replyTo.ResultQueue.Enqueue(new IsScalarResponse(value));
        replyTo.Signal.Set();
    }
    
    public static void SetShapeResponse(this ReplyTo replyTo, TensorShape value)
    {
        replyTo.ResultQueue.Enqueue(new ShapeResponse(value));
        replyTo.Signal.Set();
    }

    public static void SetEqualsResponse(this ReplyTo replyTo, bool value)
    {
        replyTo.ResultQueue.Enqueue(new EqualsResponse(value));
        replyTo.Signal.Set();
    }

    public static void SetToStringResponse(this ReplyTo replyTo, string value)
    {
        replyTo.ResultQueue.Enqueue(new ToStringResponse(value));
        replyTo.Signal.Set();
    }
}