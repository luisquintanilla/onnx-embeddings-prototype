using System.Buffers;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using TensorElementType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

internal static class TensorInteropProbe
{
    public static void Run(string fixtureDirectory)
    {
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            name = "environment", status = "observed",
            runtime = RuntimeInformation.FrameworkDescription,
            os = RuntimeInformation.OSDescription,
            architecture = RuntimeInformation.ProcessArchitecture.ToString(),
            ort = "1.23.2", tensors = "10.0.9",
            tieredCompilation = Environment.GetEnvironmentVariable("DOTNET_TieredCompilation")
        }));
        using var options = new SessionOptions { IntraOpNumThreads = 1 };
        using var session = new InferenceSession(Path.Combine(fixtureDirectory, "optional_types.onnx"), options);
        long[] data = [91, 92, 11, 12, 21, 22];
        var dense = Tensor.Create(data, [3, 2]);
        var offset = dense.Slice([1, 0]);
        var strided = Tensor.Create(data, [2, 2], [3, 1]);
        foreach (var item in new[] { ("dense", dense), ("offset", offset), ("strided", strided),
            ("singleton", Tensor.Create(data, [6, 1])), ("empty", Tensor<long>.Empty), ("shaped-empty", Tensor.CreateFromShape<long>([0, 0])) })
        {
            Report(item.Item1 + "-bridge", () =>
            {
                using var value = Bridge(item.Item2);
                return new { data = value.GetTensorDataAsSpan<long>().ToArray(), shape = value.GetTensorTypeAndShape().Shape };
            });
        }
        Report("shape-zero", () =>
        {
            var tensor = Tensor.CreateFromShape<float>([0, 0, 3]);
            return new { rank = tensor.Rank, lengths = tensor.Lengths.ToArray().Select(x => (long)x).ToArray(), flattenedLength = (long)tensor.FlattenedLength };
        });
        Report("permuted-layout", () =>
        {
            var transposed = Tensor.Create(data, [2, 2], [1, 2]);
            bool wholeSpan = transposed.TryGetSpan([0, 0], 4, out ReadOnlySpan<long> _);
            return new { transposed.IsDense, wholeSpan, logical = transposed.ToArray() };
        });
        Report("broadcast-layout", () =>
        {
            var broadcast = Tensor.Create(data, [2, 2], [0, 1]);
            bool wholeSpan = broadcast.TryGetSpan([0, 0], 4, out ReadOnlySpan<long> _);
            return new { broadcast.IsDense, wholeSpan, logical = broadcast.ToArray() };
        });
        Report("singleton-span-control", () =>
        {
            ReadOnlyTensorSpan<float> view = Tensor.Create(new float[] { 2, 6 }, [1, 2, 1]);
            bool direct = view.TryGetSpan([0, 0, 0], 2, out _);
            var squeezed = view.Squeeze().GetSpan([0], 2);
            Require(squeezed.SequenceEqual(new float[] { 2, 6 }), "Squeeze did not preserve logical singleton-axis values.");
            return new { shape = new[] { 1, 2, 1 }, directTryGetSpan = direct, squeezed = squeezed.ToArray() };
        });
        Report("stable-offset", () =>
        {
            using var value = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, data.AsMemory(2, 4), [2, 2]);
            return new { data = value.GetTensorDataAsSpan<long>().ToArray(), onnx = RunIds(session, value, 2, 2) };
        });
        Report("public-pin-offset", () => PinnedData(offset, session));
        Report("lexical-span-pin-offset", () => LexicalPin(offset, session));
        Report("lexical-span-pin-strided", () => LexicalPin(strided, session));
        Report("explicit-strided-copy", () =>
        {
            long[] copy = strided.ToArray();
            using var value = OrtValue.CreateTensorValueFromMemory(copy, [2, 2]);
            long old = data[0];
            data[0] = 999;
            try
            {
                var actual = RunIds(session, value, 2, 2);
                Require(actual.SequenceEqual(new long[] { 91, 92, 12, 21 }), "Explicit copy was not logical row-major.");
                return new { onnx = actual, sourceMutationVisible = actual[0] == 999 };
            }
            finally { data[0] = old; }
        });
        Report("stable-alias-gc", () =>
        {
            using var value = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, data.AsMemory(2, 4), [2, 2]);
            data[2] = 111;
            try
            {
                GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
                var actual = RunIds(session, value, 2, 2);
                Require(actual.SequenceEqual(new long[] { 111, 12, 21, 22 }), "Stable binding lost its pinned alias.");
                return new { onnx = actual, sourceMutationVisible = true };
            }
            finally { data[2] = 11; }
        });
        Report("stable-empty", () =>
        {
            using var value = OrtValue.CreateTensorValueFromMemory(Array.Empty<long>(), [0, 0]);
            return new { value.GetTensorTypeAndShape().Shape, elements = value.GetTensorDataAsSpan<long>().Length };
        });
        Report("stable-shape-error", () =>
        {
            using var value = OrtValue.CreateTensorValueFromMemory(new long[4], [2, 3]);
            return new { unexpected = true };
        });
        Report("stable-element-error", () =>
        {
            using var value = OrtValue.CreateTensorValueFromMemory(new float[4], [2, 2]);
            return new { unexpected = RunIds(session, value, 2, 2) };
        });
        foreach (string path in new[] { "normal", "exception", "cancellation", "construction-error" })
            Report("stable-lifetime-" + path, () => Lifetime(session, path));
        Report("binding-measurements", MeasureBindings);
    }

    private static unsafe object LexicalPin(Tensor<long> tensor, InferenceSession session)
    {
        if (tensor.Rank != 2 || !tensor.TryGetSpan([0, 0], checked((int)tensor.FlattenedLength), out ReadOnlySpan<long> span))
            throw new ArgumentException("This isolated lexical-pin probe requires a contiguous rank-two logical span.");
        fixed (long* pointer = span)
        {
            using var value = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Int64,
                [(long)tensor.Lengths[0], (long)tensor.Lengths[1]], (IntPtr)pointer, checked(span.Length * sizeof(long)));
            GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
            long[] actual = RunIds(session, value, (int)tensor.Lengths[0], (int)tensor.Lengths[1]);
            Require(actual.SequenceEqual(tensor.ToArray()), "Lexical span pin differs from logical input.");
            return new { logical = tensor.ToArray(), onnx = actual };
        }
    }

    private static object Lifetime(InferenceSession session, string path)
    {
        var counts = new PinCounts();
        var (value, weak) = CreateTracked(counts, path == "construction-error");
        if (value is not null)
        {
            try
            {
                using (value)
                {
                    GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
                    Require(weak.IsAlive, "Memory owner was not rooted for OrtValue lifetime.");
                    Require(RunIds(session, value, 2, 2).SequenceEqual(new long[] { 11, 12, 21, 22 }), "Lifetime output differs.");
                    if (path == "exception") throw new ProbeException();
                    if (path == "cancellation") new CancellationToken(canceled: true).ThrowIfCancellationRequested();
                }
            }
            catch (ProbeException) when (path == "exception") { }
            catch (OperationCanceledException) when (path == "cancellation") { }
        }
        Require(counts.Pins == counts.Unpins, "OrtValue did not release all memory pins.");
        return new { path, counts.Pins, counts.Unpins, pinnedDuringNativeUse = value is not null,
            note = "Cancellation is a deterministic lexical-cleanup probe, not an in-flight native cancellation timing test." };
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static (OrtValue? Value, WeakReference Owner) CreateTracked(PinCounts counts, bool invalidShape)
    {
        var owner = new TrackedMemory(counts);
        var weak = new WeakReference(owner);
        try
        {
            return (OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, owner.Memory, invalidShape ? [2, 3] : [2, 2]), weak);
        }
        catch (OnnxRuntimeException) when (invalidShape) { return (null, weak); }
        catch (ArgumentException) when (invalidShape) { return (null, weak); }
    }

    private sealed class PinCounts { public int Pins; public int Unpins; }
    private sealed class ProbeException : Exception;
    private sealed class TrackedMemory(PinCounts counts) : MemoryManager<long>
    {
        private readonly long[] _data = [11, 12, 21, 22];
        private GCHandle _pin;
        public override Span<long> GetSpan() => _data;
        public override unsafe MemoryHandle Pin(int elementIndex = 0)
        {
            if ((uint)elementIndex > (uint)_data.Length) throw new ArgumentOutOfRangeException(nameof(elementIndex));
            if (_pin.IsAllocated) throw new InvalidOperationException("Only one tracked pin is supported.");
            _pin = GCHandle.Alloc(_data, GCHandleType.Pinned);
            counts.Pins++;
            return new MemoryHandle((long*)_pin.AddrOfPinnedObject() + elementIndex, pinnable: this);
        }
        public override void Unpin() { _pin.Free(); counts.Unpins++; }
        protected override void Dispose(bool disposing)
        {
            if (_pin.IsAllocated) throw new InvalidOperationException("Dispose only after the OrtValue.");
        }
    }

    private static object MeasureBindings()
    {
        long[] data = Enumerable.Range(0, 32 * 256).Select(x => (long)x).ToArray();
        var tensor = Tensor.Create(data, [32, 256]);
        long[] shape = [32, 256];
        Action stable = () => { using var value = OrtValue.CreateTensorValueFromMemory(data, shape); };
        Action lexical = () => BindLexical(tensor, shape);
        for (int i = 0; i < 100; i++) { stable(); lexical(); }
        var stableRounds = new List<(double Us, long Bytes)>();
        var lexicalRounds = new List<(double Us, long Bytes)>();
        for (int round = 0; round < 7; round++)
        {
            if (round % 2 == 0) { stableRounds.Add(Measure(stable)); lexicalRounds.Add(Measure(lexical)); }
            else { lexicalRounds.Add(Measure(lexical)); stableRounds.Add(Measure(stable)); }
        }
        return new
        {
            stableUs = stableRounds.OrderBy(x => x.Us).ElementAt(3).Us,
            lexicalUs = lexicalRounds.OrderBy(x => x.Us).ElementAt(3).Us,
            stableBytes = stableRounds.OrderBy(x => x.Bytes).ElementAt(3).Bytes,
            lexicalBytes = lexicalRounds.OrderBy(x => x.Bytes).ElementAt(3).Bytes,
            note = "100 warmups, median7 alternating rounds x1000 calls; construct/dispose binding only, no inference or tensor/shape creation. Experimental bridge unavailable due to binary incompatibility. Lexical pin cannot outlive its fixed scope."
        };
    }

    private static (double Us, long Bytes) Measure(Action action)
    {
        long allocated = GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        for (int i = 0; i < 1000; i++) action();
        return (Stopwatch.GetElapsedTime(start).TotalMicroseconds / 1000, (GC.GetAllocatedBytesForCurrentThread() - allocated) / 1000);
    }

    private static unsafe void BindLexical(Tensor<long> tensor, long[] shape)
    {
        var span = tensor.GetSpan([0, 0], (int)tensor.FlattenedLength);
        fixed (long* pointer = span)
        {
            using var value = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance,
                TensorElementType.Int64, shape, (IntPtr)pointer, span.Length * sizeof(long));
        }
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }

    private static unsafe object PinnedData(Tensor<long> tensor, InferenceSession session)
    {
        using var pin = tensor.GetPinnedHandle();
        using var value = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Int64,
            [2, 2], (IntPtr)pin.Pointer, 4 * sizeof(long));
        return new
        {
            logical = tensor.ToArray(),
            firstPinned = Marshal.ReadInt64((IntPtr)pin.Pointer),
            onnx = RunIds(session, value, 2, 2)
        };
    }

    private static long[] RunIds(InferenceSession session, OrtValue ids, int batch, int sequence)
    {
        using var masks = OrtValue.CreateTensorValueFromMemory(Enumerable.Repeat(1L, batch * sequence).ToArray(), [batch, sequence]);
        using var options = new RunOptions();
        using var result = session.Run(options, new Dictionary<string, OrtValue> { ["input_ids"] = ids, ["attention_mask"] = masks }, ["last_hidden_state"]);
        var output = result[0].GetTensorDataAsSpan<float>().ToArray();
        return Enumerable.Range(0, batch * sequence).Select(i => (long)output[i * 3]).ToArray();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static OrtValue Bridge(Tensor<long> tensor)
    {
#pragma warning disable SYSLIB5001 // Isolated probe of the explicitly experimental ORT API.
        return OrtValue.CreateTensorValueFromSystemNumericsTensorObject(tensor);
#pragma warning restore SYSLIB5001
    }

    private static void Report(string name, Func<object> probe)
    {
        try { Console.WriteLine(JsonSerializer.Serialize(new { name, status = "observed", result = probe() })); }
        catch (Exception error) when (error is MissingMethodException or ArgumentException or InvalidOperationException or OnnxRuntimeException)
        {
            Console.WriteLine(JsonSerializer.Serialize(new { name, status = "failed", error = error.GetType().Name, error.Message }));
        }
    }
}
