namespace SonraML.Core.Types;

public class Parameter<T> where T : struct
{
    public Parameter(Tensor<T> value, Tensor<T> gradient, string name)
    {
        Name = name;
        Value = value;
        Gradient = gradient;
    }
    
    public string Name { get; init; }
    
    public Tensor<T> Value { get; init; }
    
    public Tensor<T> Gradient { get; init; }

    public bool HasGradient { get; private set; } = false;

    public void SetValue(Tensor<T> value)
    {
        Value.CopyFrom(value);
    }

    public void SetGradient(Tensor<T> gradient)
    {
        HasGradient = true;
        Gradient.CopyFrom(gradient);
    }

    public void ClearGradient()
    {
        HasGradient = false;
    }
}