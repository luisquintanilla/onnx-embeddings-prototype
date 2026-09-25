using Microsoft.ML.OnnxRuntime;

[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace CommunityToolkit.Embeddings.Onnx.Tests;

internal static class TestAssets
{
    public static SessionOptions Options() => new() { IntraOpNumThreads = 2, InterOpNumThreads = 1 };
    public static string Graph(string name)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", name + ".onnx");
        Assert.True(File.Exists(path), $"Missing synthetic fixture {path}. Run .venv/Scripts/python.exe dev/make_contract_graphs.py.");
        return path;
    }
    public static InferenceSession Session(string name = "optional_types")
    {
        using var options = Options();
        return new InferenceSession(Graph(name), options);
    }
    public static string Root
    {
        get
        {
            for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
                if (File.Exists(Path.Combine(directory.FullName, "OnnxEmbeddings.slnx"))) return directory.FullName;
            throw new InvalidOperationException("Run tests from the authoritative OnnxEmbeddings worktree.");
        }
    }
}
