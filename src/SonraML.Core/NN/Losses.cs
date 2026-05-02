using SonraML.Core.Interfaces;
using SonraML.Core.Types;

namespace SonraML.Core.NN;

public static class Losses
{
    public static Tensor<T> MeanSquaredError<T>
        (
        ITensorFactory tf,
        Tensor<T> prediction,
        Tensor<T> target,
        out Tensor<T> gradient
        )
        where T : struct
    {
        var diff = prediction - target;
        var loss = diff.Square().Mean();

        loss.ConvertTo<T>();

        var two = tf.Create(2).ConvertTo<T>();
        var size = tf.Create(prediction.Size).ConvertTo<T>();

        gradient = diff.Mul(two).Div(size);

        return loss;
    }
}