using SonraML.Core.Types;

namespace SonraTest.Data;

public record class MnistTrainingData(List<float[]> Inputs, List<float[]> ExpectedOutputs);