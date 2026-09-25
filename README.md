# Standalone .NET ONNX embeddings prototype

**Experimental, local inference, and unaffiliated with the .NET Community Toolkit.**
`CommunityToolkit.Embeddings.Onnx` is an experimental namespace, not an approved,
published, or supported Community Toolkit package. This is a new standalone
repository, not a Toolkit or ML.NET fork. No package is published to NuGet, and
model assets are not hosted in this repository.

Generate text embeddings entirely on your machine with explicit local model
files. One focused .NET library supplies three concrete
`IEmbeddingGenerator<string, Embedding<float>>` providers and the **same public
preparation, scoring, and postprocessing components** for direct use. There is no
ML.NET transform, `MLContext`, `IDataView`, Semantic Kernel, Agent Framework,
vector store, cloud service, model registry, or downloader in the library.

## Supported recipes

All three produce **384-dimensional, L2-normalized Float32 vectors**. The limits
include the two surrounding special tokens. Right padding uses the longest
sequence in each bounded batch, not the architecture's maximum capacity.

| Provider | Exact model | Tokenizer / padding ID | Limit | Pooling / selected ONNX output |
|---|---|---|---:|---|
| `AllMiniLmL6V2EmbeddingGenerator` | `sentence-transformers/all-MiniLM-L6-v2` | Uncased BERT WordPiece / 0 | 256 | Masked mean / `last_hidden_state` |
| `E5SmallV2EmbeddingGenerator` | `intfloat/e5-small-v2` | Uncased BERT WordPiece / 0 | 512 | Masked mean / `last_hidden_state` |
| `GraniteEmbedding30MEnglishGenerator` | `ibm-granite/granite-embedding-30m-english` | RoBERTa byte-level BPE / 1 | 512 | CLS / `logits` |

These are variant-specific providers, not general E5 or Granite family support.
MiniLM's recipe limit is **256**, despite BERT's architectural capacity of 512.
BERT lowercasing and accent handling follow `tokenizer_config.json`, not the
separate `sentence_bert_config.json` lowercasing field. Granite's 514 position
embeddings do **not** mean 514 input tokens. Its normal embedding recipe does
not add the separate r1.1 conversational retrieval format.

The tested publisher revisions are pinned in [`dev/models.json`](dev/models.json):

| Model | Revision |
|---|---|
| MiniLM | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| E5 | `ffb93f3bd4047442299a41ebb6fa998a38507c52` |
| Granite | `9b5b096411652ec1189c68fcfb90d0a82c5b45af` |

## Start here

Install the **.NET SDK 10.0.401** pinned by `global.json`. Run the commands from
this repository's root in PowerShell. Normal build/restore obtains NuGet
dependencies, **never model files**.

```powershell
dotnet restore .\OnnxEmbeddings.slnx --locked-mode
dotnet build .\OnnxEmbeddings.slnx --no-restore
```

The repository-scoped `NuGet.Config` uses Microsoft's public `dotnet-public`
mirror: this development environment could reach that feed, but
`api.nuget.org` failed TLS negotiation. No global NuGet settings, certificate
validation, or signature checks were changed. SDK, direct dependencies, and
resolved transitive graphs are pinned. Internet access is needed for a first
restore unless the packages are already cached.

### Supply local assets

Applications choose acquisition, licensing, mirrors, storage, and update policy.
The library does not download anything during construction, inference, or
disposal, and does not require hashes or a runtime network allowlist.

Each model directory needs these files:

```text
minilm\                        e5\                         granite\
  model.onnx                     model.onnx                 model.onnx
  vocab.txt                      vocab.txt                  vocab.json
                                                           merges.txt
```

Use trusted, mutually compatible weights and vocabularies. The verified files
are the FP32 publisher exports at the revisions above: MiniLM's
`onnx/model.onnx` (renamed locally to `model.onnx`), and E5/Granite's `model.onnx`.
Different exports, quantization, or revisions need their own verification.
Model licenses are separate from this prototype: publisher metadata identifies
MiniLM and Granite as Apache-2.0 and E5 as MIT; review the applicable model cards
and your organization's policies yourself.

For a reproducible **developer-only** download of this bounded set, use Python
3.12 and the explicit script below. This also downloads the corresponding
`model.safetensors` and configuration files for independent verification.
Expect roughly 0.7 GB of model assets plus the isolated Python environment.
No remote model code is executed; `trust_remote_code=False` and
`local_files_only=True` are used for reference loading.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\dev\requirements.lock.txt
.\.venv\Scripts\python.exe .\dev\acquire.py
.\.venv\Scripts\python.exe .\dev\reference.py
```

`acquire.py` is the **only model-network step**. It records source URLs,
revisions, sizes, and SHA-256 values in each ignored `.assets\<model>\provenance.json`.
It checks files against the committed validation hashes when available; this is
a developer verification guard, not a runtime model allowlist. Existing files
are not silently replaced. `reference.py` is offline: it checks
ONNX contracts, compares publisher ONNX outputs to independently loaded
HF/PyTorch Float32 weights, and produces tokenizer and final-vector goldens.
No Python installation is needed by .NET consumers or the samples.
`requirements.txt` identifies the direct development tools;
`requirements.lock.txt` pins the complete verified Windows/Python 3.12 dependency
graph, including the independent HF `tokenizers` reference engine.

### Run the provider sample

```powershell
dotnet run --project .\samples\Providers -- minilm .\.assets\minilm
dotnet run --project .\samples\Providers -- e5 .\.assets\e5
dotnet run --project .\samples\Providers -- granite .\.assets\granite
```

This uses `IEmbeddingGenerator<string, Embedding<float>>`, scores a database
search query against three passages, and prints a ranked list. E5 uses the
explicit mixed-purpose overload for the query and documents. Scores are
model-dependent; do not compare their numerical values across model families.

```csharp
using CommunityToolkit.Embeddings.Onnx;
using Microsoft.Extensions.AI;
using Microsoft.ML.OnnxRuntime;

using var sessionOptions = new SessionOptions { IntraOpNumThreads = 2 };
using IEmbeddingGenerator<string, Embedding<float>> generator =
    new AllMiniLmL6V2EmbeddingGenerator(localDirectory, sessionOptions);
var embeddings = await generator.GenerateAsync(["A puppy plays outside.", "A dog is outdoors."]);
ReadOnlyMemory<float> firstVector = embeddings[0].Vector;
```

### E5 query/document purpose

E5 construction requires `E5Purpose.Query` or `E5Purpose.Document`. Its
`IEmbeddingGenerator<string, Embedding<float>>` method applies that purpose to
**every item** in the call. For asymmetric retrieval, queries use `query: ` and
documents use `passage: `. For a batch with both, the concrete provider also
accepts `IEnumerable<E5Input>`:

```csharp
using var e5 = new E5SmallV2EmbeddingGenerator(localDirectory, E5Purpose.Query);
var vectors = await e5.GenerateAsync(new E5Input[]
{
    new("How do indexes help?", E5Purpose.Query),
    new("An index avoids scanning every row.", E5Purpose.Document)
});
```

The prefix is always prepended exactly once by the provider; caller text is
never heuristically stripped or trimmed. Thus input already beginning with
`query: ` becomes `query: query: ...`. Supply **unprefixed** text to providers.
Direct-stage callers instead invoke the public `E5Text.Format` themselves.
No fictional MEAI purpose option is used.

## Public composition, without an embedding framework

```text
TextTokenizer -> TextBatchPreparer -> TokenBatch
                                         |
                               OnnxTextEncoder.Score
                                         |
                              EmbeddingPooling.Pool
                                         |
                                  float[][] vectors
```

`TextTokenizer` combines a content-token callable with BOS/EOS/pad IDs.
`TextBatchPreparer` applies single-sequence special tokens, content truncation,
Int32-to-Int64 conversion, optional zero token types, and padding. `TokenBatch`
owns flat row-major managed arrays exposed as read-only memories.
`OnnxTextEncoder` validates and executes an explicit token-level ONNX graph.
`EmbeddingPooling` exposes masked mean, CLS, and stable in-place normalization.
No stage requires an MEAI or ML.NET data type.

The **independent assembly** in [`samples/Composition`](samples/Composition)
proves that these are public APIs, not an internal parallel implementation:

```powershell
dotnet run --project .\samples\Composition -- minilm .\.assets\minilm
dotnet run --project .\samples\Composition -- e5 .\.assets\e5
dotnet run --project .\samples\Composition -- granite .\.assets\granite
```

It builds a preparer, calls `encoder.Score(batch)`, then
`EmbeddingPooling.Pool(hidden, batch, encoder.Dimensions, pooling)`, exactly as
the providers do. Applications can also compose `OnnxEmbeddingGenerator`
directly from these stages. This is ordinary composition, not a general-purpose
pipeline, plugin system, or model manifest platform.

## Contracts, resources, and runtime selection

The library references **`Microsoft.ML.OnnxRuntime.Managed` only**. Each executable
sample, experiment, or test project explicitly selects the matching
`Microsoft.ML.OnnxRuntime` **1.23.2 CPU** native package. Consumers must select
one compatible native runtime; do not blindly combine CPU, GPU, and WinML
packages. GPU/WinML execution, I/O binding, and acceleration are **not verified
or claimed** here. Options and session injection permit deliberate configuration.

| Resource / API | Contract |
|---|---|
| `OnnxTextEncoder.Load(path, options)` | Owns its newly created session; borrows options only during construction. Disposes the session if validation fails. |
| `new OnnxTextEncoder(session, ..., ownsSession: false)` | Borrows the supplied session by default. Ownership transfers only on successful construction when explicitly requested. |
| Concrete providers with a directory | Own their encoder/session; do not dispose caller options. |
| Concrete providers with an existing session | Own the wrapper, but borrow the native session unless `ownsSession: true`. |
| `new OnnxEmbeddingGenerator(preparer, encoder, ...)` | Borrows the encoder by default; explicit `ownsEncoder: true` transfers disposal responsibility. |
| `TokenBatch`, returned arrays / embeddings | Ordinary GC-owned buffers; valid after the call and after provider disposal. Public batch construction copies supplied spans. |
| `Score` native handles | Per-call `OrtValue` inputs pin managed Int64 buffers; outputs are copied once into an owned managed array, then native handles are disposed. This is **not end-to-end zero-copy**. |

Synchronous local inference returns a completed `Task` through MEAI; there is
no implicit thread-pool scheduling. Cancellation is checked before work,
between texts/batches, and after inference; a per-call `RunOptions.Terminate`
registration cancels active native inference. A single tokenizer operation is
not interruptible. Cancellation is reported as cancellation, not empty output.

Concurrent CPU inference calls share the session and use independent per-call
buffers/options, without a global inference lock. Respect the thread safety of
any caller-supplied tokenizer callable and execution provider. **Do not dispose
the generator, encoder, session, or mutate caller configuration while calls are
active**; the caller must await/join active work before disposal. No concurrency
guarantee for unverified execution providers is implied.

An empty input sequence is a legitimate empty result; empty/whitespace **text**
is tokenized and embedded. Nulls, malformed UTF-16, invalid batch/mask shapes, unsupported options,
and incompatible graphs fail explicitly. Inputs must be Int64 rank-2
`input_ids`, `attention_mask`, and optionally `token_type_ids`; unexpected inputs
are rejected. Static graph dimensions are checked against each batch. The
selected output must be Float32 rank-3 `[batch, sequence, 384]`. There is no
"first output" fallback and no pooling of rank-2 already-pooled tensors.

Granite's token output happens to be called **`logits`**; the inspected graph
also has a rank-2 output named `894`, which is never requested. The output
contract is based on inspected semantics and PyTorch parity, not the name.

Mean pooling rejects all-padding rows. CLS requires an unmasked first token.
Pooling rejects nonfinite values, including masked positions; normalization
rejects zero/nonfinite vectors and scales before computing the norm to avoid
overflow/underflow. Pooling arithmetic overflow is an explicit error. The fixed
MEAI model ID/dimensions must match; unrecognized additional/raw options throw.
Metadata is available via `GetService(typeof(EmbeddingGeneratorMetadata))`.

## Verification, measurements, and limits

```powershell
# Fast, self-contained unit/contract tests; no downloaded assets or Python needed.
dotnet test .\tests\CommunityToolkit.Embeddings.Onnx.Tests --filter "Category!=RealModels"

# Full verification, after acquiring local assets and generating references above.
dotnet test .\OnnxEmbeddings.slnx
```

The complete run passes **95 tests with zero skips**, including exact token
comparisons and final-vector comparisons over **139 independent reference rows**
for all three models. Missing real-model assets cause actionable failures,
not silently skipped integration tests.

See [`docs/VALIDATION.md`](docs/VALIDATION.md) for reproducible commands, pinned
reference evidence, exact token/mask comparisons, final-vector tolerances,
observed errors, dependency inventory, and the RoBERTa reproduction.

The library has four direct dependencies: `Microsoft.ML.Tokenizers` 2.0.0,
`System.Numerics.Tensors` 10.0.9, `Microsoft.ML.OnnxRuntime.Managed` 1.23.2,
and `Microsoft.Extensions.AI.Abstractions` 10.3.0. A single library intentionally
means direct-stage users still receive the MEAI assembly transitively, even
though the stage signatures do not depend on it. There is no justified separate
abstractions package yet.

This is a CPU correctness prototype, not an optimized production serving stack.
Tokenization currently materializes content tokens **before truncating** to
preserve exact HF token-boundary semantics (including partially retained
multi-token Unicode characters). Extremely large individual strings still cost
CPU/memory proportional to their length; applications should enforce their own
request-size limits. Inference batches default to at most 32, not an unlimited
native batch. No text pairs, arbitrary tokenizer JSON loader, quantized graphs,
trained-model updates, network serving, streaming embeddings, or ML.NET adapters
are included. English models are not a multilingual quality guarantee merely
because Unicode tokenization is tested.

### What might belong upstream

- **Tokenizers, correctness candidates:** the reproduced BERT basic-text
  discrepancies (discarded astral/symbol text, tab boundaries, decomposition and
  normalized added-token handling). The local compatibility adapter still uses
  Microsoft's WordPiece engine; correct ordinary examples were not enough to
  establish parity. These are scoped repros against pinned HF recipes, not a
  demand to change every tokenizer's semantics.
- **Tokenizers, potential proposal:** a supported byte-level RoBERTa composition
  with configurable HF added-token whitespace semantics and exact token-budget
  truncation. `EnglishRobertaTokenizer` cannot be substituted by name alone;
  the reproducible comparison distinguishes its Fairseq mapping, byte
  preprocessing, and literal special-token behavior. This library reuses
  Microsoft's `BpeTokenizer`; it does not implement a second BPE engine.
- **Tokenizers, potential proposal:** caller-buffer / batch ID-and-mask
  preparation that avoids intermediate lists and widening copies. The focused
  harness measures complete preparation versus a straightforward allocating
  baseline, not token counting. It does not establish that a proposed upstream
  API would achieve a particular speedup.
- **Tensors:** existing `TensorPrimitives` add, divide, norm and cosine operations
  suffice here. Measurements do not justify a new general numerical API, nor
  an automatic zero-copy ORT integration claim.
- **Keep local/model-specific:** E5 purpose and prefixes, model sequence limits,
  BOS/EOS recipes, pooling selection, ONNX output names, and whether embeddings
  should be normalized. These are not general tensor-domain semantics.

No upstream change or submission is part of this milestone.
