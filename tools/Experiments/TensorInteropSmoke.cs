#if INTEROP_SMOKE
using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;

if (args.Length != 1) throw new ArgumentException("Supply the synthetic fixture directory.");
using var session = new InferenceSession(Path.Combine(args[0], "optional_types.onnx"));
long[] source = [91, 92, 11, 12, 21, 22];
Tensor<long> tensor = Tensor.Create(source, [3, 2]).Slice([1, 0]);
if (!tensor.GetSpan([0, 0], 4).SequenceEqual(new long[] { 11, 12, 21, 22 }))
    throw new InvalidOperationException("Logical tensor slice differs.");
using var ids = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, source.AsMemory(2, 4), [2, 2]);
using var masks = OrtValue.CreateTensorValueFromMemory(new long[] { 1, 1, 1, 1 }, [2, 2]);
using var options = new RunOptions();
using var output = session.Run(options, new Dictionary<string, OrtValue> { ["input_ids"] = ids, ["attention_mask"] = masks }, ["last_hidden_state"]);
Tensor<float> owned = Tensor.Create(output[0].GetTensorDataAsSpan<float>().ToArray(), [2, 2, 3]);
if (owned[0, 0, 0] != 11 || owned[0, 1, 0] != 12 || owned[1, 0, 0] != 21 || owned[1, 1, 0] != 22)
    throw new InvalidOperationException("Native output differs from the stable offset control.");
Console.WriteLine("PASS: shaped offset input, stable Memory ORT binding, owned shaped output [2,2,3].");
#endif
