using System.Collections.Concurrent;
using SonraML.Backend.MLX.ExecutionManagement;
using SonraML.Backend.MLX.Interop.Enums;
using SonraML.Core.Exceptions;
using SonraML.Core.Types;

namespace SonraML.Backend.MLX.Implementations;

internal class ProxyTensor<T> : Tensor<T> where T : struct
{
    private readonly MlxTensorFactory tf;
    private readonly ConcurrentQueue<MlxCommand> commandQueue;
    private readonly ConcurrentQueue<MlxResponse> responseQueue = new();
    private readonly EventWaitHandle opDone = new(false, EventResetMode.AutoReset);
    private readonly SemaphoreSlim tlock = new(1, 1);

    public ProxyTensor(Guid id, MlxTensorFactory tf, ConcurrentQueue<MlxCommand> commandQueue, string? name)
    {
        Id = id;
        this.tf = tf;
        this.commandQueue = commandQueue;
        Name = name ?? Id.ToString();
    }
    
    #region Properties

    public Guid Id { get; }

    private ReplyTo ReplyTo => new(responseQueue, opDone);

    public override TensorShape Shape
    {
        get
        {
            tlock.Wait();

            commandQueue.Enqueue(new GetShapeTensorOp(ReplyTo, Id));
            opDone.WaitOne();

            var sr = GetResponse<ShapeResponse>();

            tlock.Release();
            return sr.Shape;
        }
    }

    public override bool IsScalar
    {
        get
        {
            tlock.Wait();

            commandQueue.Enqueue(new IsScalarTensorOp(ReplyTo, Id));
            opDone.WaitOne();

            var ic = GetResponse<IsScalarResponse>();

            tlock.Release();
            return ic.IsScalar;
        }
    }
    
    #endregion

    #region ObjectMethods

    public override IEnumerator<T> GetEnumerator()
    {
        var type = MlxDType.GetDType<T>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(T));
        }
        
        tlock.Wait();

        commandQueue.Enqueue(new GetEnumeratorTensorOp(ReplyTo, Id, type.Value));
        opDone.WaitOne();

        if (!responseQueue.TryDequeue(out var result))
        {
            tlock.Release();
            throw new BackendOperationException("Expected MLX Response.");
        }
        
        if (result is ErrorResponse error)
        {
            tlock.Release();
            throw new BackendOperationException(error.Message);
        }

        if (result is not EnumeratorResponse er)
        {
            tlock.Release();
            throw new BackendOperationException($"Expected {nameof(EnumeratorResponse)}.");
        }

        tlock.Release();
        return (IEnumerator<T>)er.Enumerator;
    }

    public override object Clone()
    {
        tlock.Wait();

        commandQueue.Enqueue(new CopyTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override void CopyFrom(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new CopyFromTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        GetResponse<SuccessResponse>();

        tlock.Release();
    }

    public override Tensor<TTarget> ConvertTo<TTarget>()
    {
        var type = MlxDType.GetDType<TTarget>();
        if (type is null)
        {
            throw new TensorTypeNotSupportedException(typeof(TTarget));
        }
        
        tlock.Wait();

        commandQueue.Enqueue(new ConvertTensorOp(ReplyTo, Id, type.Value));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<TTarget>(tr.Id, tr.Type);
    }

    public override string ToString()
    {
        tlock.Wait();
        
        commandQueue.Enqueue(new ToStringOp(ReplyTo, Id));
        opDone.WaitOne();
        
        if (!responseQueue.TryDequeue(out var result))
        {
            tlock.Release();
            throw new BackendOperationException("Expected MLX Response.");
        }
        
        if (result is ErrorResponse error)
        {
            tlock.Release();
            throw new BackendOperationException(error.Message);
        }
    
        if (result is not ToStringResponse stringResponse)
        {
            tlock.Release();
            throw new BackendOperationException($"Expected {nameof(EnumeratorResponse)}.");
        }
        
        tlock.Release();
        return stringResponse.Value;
    }
    
    #endregion
    
    #region TensorOps

    public override void EnsureCompute()
    {
        tlock.Wait();

        commandQueue.Enqueue(new EnsureComputeTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        GetResponse<SuccessResponse>();

        tlock.Release();
    }
    
    #region ArithmeticOps

    public override Tensor<T> Add(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new AddTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Sub(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new SubTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Mul(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new MulTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Rec()
    {
        tlock.Wait();

        commandQueue.Enqueue(new RecTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Div(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new DivTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Mod(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new ModTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Rem(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new RemTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Neg()
    {
        tlock.Wait();

        commandQueue.Enqueue(new NegTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Abs()
    {
        tlock.Wait();

        commandQueue.Enqueue(new AbsTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Sign()
    {
        tlock.Wait();

        commandQueue.Enqueue(new SignTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #region LogicalOps

    public override Tensor<bool> Equal(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new EqualTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> NotEqual(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new NotEqualTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> Less(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new LessTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> LessEqual(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new LessEqualTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> Greater(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new GreaterTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> GreaterEqual(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new GreaterEqualTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> And(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new AndTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> Or(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new OrTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> Not()
    {
        tlock.Wait();

        commandQueue.Enqueue(new NotTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> IsNAN()
    {
        tlock.Wait();

        commandQueue.Enqueue(new IsNANTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> IsInfinity()
    {
        tlock.Wait();

        commandQueue.Enqueue(new IsInfinityTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> IsFinite()
    {
        tlock.Wait();

        commandQueue.Enqueue(new IsFiniteTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> IsClose(Tensor<T> other, double rTol, double aTol, bool equalNAN = false)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new IsCloseTensorOp(ReplyTo, Id, o.Id, rTol, aTol, equalNAN));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }

    public override Tensor<bool> AllClose(Tensor<T> other, double rTol, double aTol, bool equalNAN = false)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new AllCloseTensorOp(ReplyTo, Id, o.Id, rTol, aTol, equalNAN));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<bool>(tr.Id, tr.Type);
    }
    
    #endregion

    #region BitwiseOps

    public override Tensor<T> BitwiseAnd(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new BitwiseAndTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> BitwiseOr(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new BitwiseOrTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> BitwiseXor(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new BitwiseXorTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> BitwiseNot()
    {
        tlock.Wait();

        commandQueue.Enqueue(new BitwiseNotTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #region ExponentialOps

    public override Tensor<T> Exp()
    {
        tlock.Wait();

        commandQueue.Enqueue(new ExpTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ExpM1()
    {
        tlock.Wait();

        commandQueue.Enqueue(new ExpM1TensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Log()
    {
        tlock.Wait();

        commandQueue.Enqueue(new LogTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Log10()
    {
        tlock.Wait();

        commandQueue.Enqueue(new Log10TensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Log2()
    {
        tlock.Wait();

        commandQueue.Enqueue(new Log2TensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Log1P()
    {
        tlock.Wait();

        commandQueue.Enqueue(new Log1PTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Square()
    {
        tlock.Wait();

        commandQueue.Enqueue(new SquareTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Sqrt()
    {
        tlock.Wait();

        commandQueue.Enqueue(new SqrtTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> RSqrt()
    {
        tlock.Wait();

        commandQueue.Enqueue(new RSqrtTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Pow(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new PowTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> LogSumExp(bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new LogSumExpTensorOp(ReplyTo, Id, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> LogSumExp(int axis, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new LogSumExpAxisTensorOp(ReplyTo, Id, axis, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> LogSumExp(int[] axes, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new LogSumExpAxesTensorOp(ReplyTo, Id, axes, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> LogAddExp(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new LogAddExpTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #region TrigonometricOps

    public override Tensor<T> Sin()
    {
        tlock.Wait();

        commandQueue.Enqueue(new SinTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> SinH()
    {
        tlock.Wait();

        commandQueue.Enqueue(new SinHTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ArcSin()
    {
        tlock.Wait();

        commandQueue.Enqueue(new ArcSinTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ArcSinH()
    {
        tlock.Wait();

        commandQueue.Enqueue(new ArcSinHTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Cos()
    {
        tlock.Wait();

        commandQueue.Enqueue(new CosTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> CosH()
    {
        tlock.Wait();

        commandQueue.Enqueue(new CosHTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ArcCos()
    {
        tlock.Wait();

        commandQueue.Enqueue(new ArcCosTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ArcCosH()
    {
        tlock.Wait();

        commandQueue.Enqueue(new ArcCosHTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Tan()
    {
        tlock.Wait();

        commandQueue.Enqueue(new TanTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> TanH()
    {
        tlock.Wait();

        commandQueue.Enqueue(new TanHTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ArcTan()
    {
        tlock.Wait();

        commandQueue.Enqueue(new ArcTanTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ArcTanH()
    {
        tlock.Wait();

        commandQueue.Enqueue(new ArcTanHTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ArcTan2(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new ArcTan2TensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #region Rounding

    public override Tensor<T> Floor()
    {
        tlock.Wait();

        commandQueue.Enqueue(new FloorTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Round(int decimals)
    {
        tlock.Wait();

        commandQueue.Enqueue(new RoundTensorOp(ReplyTo, Id, decimals));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Ceil()
    {
        tlock.Wait();

        commandQueue.Enqueue(new CeilTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Clip(T min, T max)
    {
        tlock.Wait();

        var minT = tf.Create(min) as ProxyTensor<T> ?? throw new InvalidOperationException();
        var maxT = tf.Create(max) as ProxyTensor<T> ?? throw new InvalidOperationException();

        commandQueue.Enqueue(new ClipTensorOp(ReplyTo, Id, minT.Id, maxT.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        commandQueue.Enqueue(new DeleteManyOp(ReplyTo, [minT.Id, maxT.Id]));
        opDone.WaitOne();

        GetResponse<SuccessResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> FloorDiv(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new FloorDivTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #region MatrixOps

    public override Tensor<T> MatMul(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new MatMulTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Fma(Tensor<T> b, Tensor<T> c, float alpha = 1, float beta = 1)
    {
        tlock.Wait();
        if (b is not ProxyTensor<T> ptB)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        if (c is not ProxyTensor<T> ptC)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new FmaTensorOp(ReplyTo, Id, ptB.Id, ptC.Id, alpha, beta));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Transpose()
    {
        tlock.Wait();

        commandQueue.Enqueue(new TransposeTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Transpose(int[] axes)
    {
        tlock.Wait();

        commandQueue.Enqueue(new TransposeAxesTensorOp(ReplyTo, Id, axes));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> SwapAxes(int a, int b)
    {
        tlock.Wait();

        commandQueue.Enqueue(new SwapAxesTensorOp(ReplyTo, Id, a, b));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> MoveAxis(int src, int dest)
    {
        tlock.Wait();

        commandQueue.Enqueue(new MoveAxisTensorOp(ReplyTo, Id, src, dest));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #region ShapeOps

    public override Tensor<T> Diag(int diagonal)
    {
        tlock.Wait();

        commandQueue.Enqueue(new DiagTensorOp(ReplyTo, Id, diagonal));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Reshape(TensorShape shape)
    {
        tlock.Wait();

        commandQueue.Enqueue(new ReshapeTensorOp(ReplyTo, Id, shape));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Flatten(int startAxis, int endAxis)
    {
        tlock.Wait();

        commandQueue.Enqueue(new FlattenTensorOp(ReplyTo, Id, startAxis, endAxis));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ExpandDims(int axis)
    {
        tlock.Wait();

        commandQueue.Enqueue(new ExpandDimsTensorOp(ReplyTo, Id, axis));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ExpandDims(int[] axes)
    {
        tlock.Wait();

        commandQueue.Enqueue(new ExpandDimsAxesTensorOp(ReplyTo, Id, axes));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> BroadcastTo(TensorShape shape)
    {
        tlock.Wait();

        commandQueue.Enqueue(new BroadcastToTensorOp(ReplyTo, Id, shape));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #region IndexingOps

    public override Tensor<T> Slice(int[] start, int[] stop, int[] strides)
    {
        tlock.Wait();

        commandQueue.Enqueue(new SliceTensorOp(ReplyTo, Id, start, stop, strides));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> DynamicSlice(Tensor<T> start, int[] axes, int[] sliceSize)
    {
        tlock.Wait();
        if (start is not ProxyTensor<T> s)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new DynamicSliceTensorOp(ReplyTo, Id, s.Id, axes, sliceSize));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> SliceUpdate(Tensor<T> update, int[] start, int[] stop, int[] strides)
    {
        tlock.Wait();
        if (update is not ProxyTensor<T> u)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new SliceUpdateTensorOp(ReplyTo, Id, u.Id, start, stop, strides));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Take(Tensor<T> indices)
    {
        tlock.Wait();
        if (indices is not ProxyTensor<T> i)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new TakeTensorOp(ReplyTo, Id, i.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Take(Tensor<T> indices, int axis)
    {
        tlock.Wait();
        if (indices is not ProxyTensor<T> i)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new TakeAxisTensorOp(ReplyTo, Id, i.Id, axis));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> TakeAlongAxis(Tensor<T> indices, int axis)
    {
        tlock.Wait();
        if (indices is not ProxyTensor<T> i)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new TakeAlongAxisTensorOp(ReplyTo, Id, i.Id, axis));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Gather(Tensor<T>[] indices, int[] axes, int[] sliceSices)
    {
        tlock.Wait();
        var i = indices.Cast<ProxyTensor<T>>().Select(t => t.Id).ToList();
        if (i.Count != indices.Length)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new GatherTensorOp(ReplyTo, Id, i, axes, sliceSices));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #region SplitOps

    public override Tensor<T>[] Split(int numSplits, int axis)
    {
        tlock.Wait();

        commandQueue.Enqueue(new SplitTensorOp(ReplyTo, Id, numSplits, axis));
        opDone.WaitOne();

        var tr = GetResponse<TensorArrayResponse>();

        tlock.Release();
        return tr
            .Id
            .Select(id => tf.CreateProxyTensor<T>(id, tr.Type))
            .Cast<Tensor<T>>()
            .ToArray();
    }

    public override Tensor<T>[] Split(int[] indices, int axis)
    {
        tlock.Wait();

        commandQueue.Enqueue(new SplitIndicesTensorOp(ReplyTo, Id, indices, axis));
        opDone.WaitOne();

        var tr = GetResponse<TensorArrayResponse>();

        tlock.Release();
        return tr
            .Id
            .Select(id => tf.CreateProxyTensor<T>(id, tr.Type))
            .Cast<Tensor<T>>()
            .ToArray();
    }
    
    #endregion

    #region PredicateOps

    public override Tensor<T> Sum(bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new SumTensorOp(ReplyTo, Id, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Sum(int axis, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new SumAxisTensorOp(ReplyTo, Id, axis, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Sum(int[] axes, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new SumAxesTensorOp(ReplyTo, Id, axes, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Min(bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new MinTensorOp(ReplyTo, Id, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Min(int axis, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new MinAxisTensorOp(ReplyTo, Id, axis, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Min(int[] axes, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new MinAxesTensorOp(ReplyTo, Id, axes, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Max(bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new MaxTensorOp(ReplyTo, Id, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Max(int axis, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new MaxAxisTensorOp(ReplyTo, Id, axis, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Max(int[] axes, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new MaxAxesTensorOp(ReplyTo, Id, axes, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Mean(bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new MeanTensorOp(ReplyTo, Id, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Mean(int axis, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new MeanAxisTensorOp(ReplyTo, Id, axis, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Mean(int[] axes, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new MeanAxesTensorOp(ReplyTo, Id, axes, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Std(int ddof, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new StdTensorOp(ReplyTo, Id, ddof, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Std(int axis, int ddof, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new StdAxisTensorOp(ReplyTo, Id, axis, ddof, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Std(int[] axes, int ddof, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new StdAxesTensorOp(ReplyTo, Id, axes, ddof, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ArgMin(bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new ArgMinTensorOp(ReplyTo, Id, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ArgMin(int axis, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new ArgMinAxisTensorOp(ReplyTo, Id, axis, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ArgMax(bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new ArgMaxTensorOp(ReplyTo, Id, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ArgMax(int axis, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new ArgMaxAxisTensorOp(ReplyTo, Id, axis, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Variance(bool keepDims, int ddof)
    {
        tlock.Wait();

        commandQueue.Enqueue(new VarianceTensorOp(ReplyTo, Id, ddof, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #region SelectionOps

    public override Tensor<T> All(bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new AllTensorOp(ReplyTo, Id, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> All(int axis, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new AllAxisTensorOp(ReplyTo, Id, axis, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> All(int[] axes, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new AllAxesTensorOp(ReplyTo, Id, axes, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Any(bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new AnyTensorOp(ReplyTo, Id, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Any(int axis, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new AnyAxisTensorOp(ReplyTo, Id, axis, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Any(int[] axes, bool keepDims = false)
    {
        tlock.Wait();

        commandQueue.Enqueue(new AnyAxesTensorOp(ReplyTo, Id, axes, keepDims));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<TResult> Where<TResult>(Tensor<TResult> ifTrue, Tensor<TResult> ifFalse)
    {
        if (ifTrue is not ProxyTensor<TResult> ifT)
        {
            throw new TensorCompatibilityException();
        }

        if (ifFalse is not ProxyTensor<TResult> ifF)
        {
            throw new TensorCompatibilityException();
        }
        
        tlock.Wait();

        commandQueue.Enqueue(new WhereTensorOp(ReplyTo, Id, ifT.Id, ifF.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<TResult>(tr.Id, tr.Type);
    }

    public override Tensor<T> Minimum(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new MinimumTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Maximum(Tensor<T> other)
    {
        tlock.Wait();
        if (other is not ProxyTensor<T> o)
        {
            tlock.Release();
            throw new TensorCompatibilityException();
        }

        commandQueue.Enqueue(new MaximumTensorOp(ReplyTo, Id, o.Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> TopK(int k)
    {
        tlock.Wait();

        commandQueue.Enqueue(new TopKTensorOp(ReplyTo, Id, k));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> TopK(int k, int axis)
    {
        tlock.Wait();

        commandQueue.Enqueue(new TopKAxisTensorOp(ReplyTo, Id, k, axis));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #region LikeOps

    public override Tensor<T> ZerosLike()
    {
        tlock.Wait();

        commandQueue.Enqueue(new ZerosLikeTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> OnesLike()
    {
        tlock.Wait();

        commandQueue.Enqueue(new OnesLikeTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #region NeuralOps

    public override Tensor<T> Sigmoid()
    {
        tlock.Wait();

        commandQueue.Enqueue(new SigmoidTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Softmax(bool precise = true)
    {
        tlock.Wait();

        commandQueue.Enqueue(new SoftmaxTensorOp(ReplyTo, Id, precise));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Softmax(int axis, bool precise = true)
    {
        tlock.Wait();

        commandQueue.Enqueue(new SoftmaxAxisTensorOp(ReplyTo, Id, axis, precise));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Softmax(int[] axes, bool precise = true)
    {
        tlock.Wait();

        commandQueue.Enqueue(new SoftmaxAxesTensorOp(ReplyTo, Id, axes, precise));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> Erf()
    {
        tlock.Wait();

        commandQueue.Enqueue(new ErfTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }

    public override Tensor<T> ErfInv()
    {
        tlock.Wait();

        commandQueue.Enqueue(new ErfInvTensorOp(ReplyTo, Id));
        opDone.WaitOne();

        var tr = GetResponse<TensorResponse>();

        tlock.Release();
        return tf.CreateProxyTensor<T>(tr.Id, tr.Type);
    }
    
    #endregion

    #endregion

    private TResponse GetResponse<TResponse>() where TResponse : MlxResponse
    {
        if (!responseQueue.TryDequeue(out var response))
        {
            tlock.Release();
            throw new BackendOperationException("Expected MLX Response.");
        }
        
        if (response is ErrorResponse error)
        {
            tlock.Release();
            throw new BackendOperationException(error.Message);
        }
        
        if (response is not TResponse tr)
        {
            tlock.Release();
            throw new BackendOperationException($"Expected {typeof(TResponse).Name}.");
        }

        return tr;
    }
}