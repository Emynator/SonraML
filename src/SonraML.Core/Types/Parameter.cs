namespace SonraML.Core.Types;

/// <summary>
/// Parameter class for NNModule parameters.
/// </summary>
/// <typeparam name="T">DType of param tensor.</typeparam>
public class Parameter<T> where T : struct
{
    /// <summary>
    /// ctor.
    /// </summary>
    /// <param name="value">See property Name.</param>
    /// <param name="gradient">See property Value.</param>
    /// <param name="name">See property Gradient.</param>
    public Parameter(Tensor<T> value, Tensor<T> gradient, string name)
    {
        Name = name;
        Value = value;
        Gradient = gradient;
    }
    
    /// <summary>
    /// Parameter name - relevant for saving and loading.
    /// </summary>
    public string Name { get; init; }
    
    /// <summary>
    /// Tensor of the actual weights stored in the param.
    /// </summary>
    public Tensor<T> Value { get; init; }
    
    /// <summary>
    /// Tensor of the gradients, relevant for backprop and optimizers.
    /// </summary>
    public Tensor<T> Gradient { get; init; }

    /// <summary>
    /// Have gradients been set an optimizer can use?
    /// </summary>
    public bool HasGradient { get; private set; } = false;

    /// <summary>
    /// Setter for value, used by optimizer to update weights.
    /// </summary>
    /// <param name="value">Updated weight value.</param>
    public void SetValue(Tensor<T> value)
    {
        Value.CopyFrom(value);
    }

    /// <summary>
    /// Sets gradients for optimizer to use.
    /// </summary>
    /// <param name="gradient">Gradient value.</param>
    public void SetGradient(Tensor<T> gradient)
    {
        HasGradient = true;
        Gradient.CopyFrom(gradient);
    }

    /// <summary>
    /// Clears gradients after optimizer run.
    /// </summary>
    public void ClearGradient()
    {
        HasGradient = false;
    }
}