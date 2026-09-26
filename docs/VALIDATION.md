# Validation and reproducibility

This document distinguishes **publisher export validation** (Python ONNX versus
independent PyTorch weights) from **.NET provider validation** (Microsoft
tokenization, public stages and concrete providers versus independent goldens).
Ranking or compatible tensor shapes alone do not establish parity.

## Verified result

**213 passed, 0 failed, 0 skipped**: 199 deterministic contract/regression cases and
14 real-model test cases. All 157 published tokenizer-refactor cases remain,
with 56 new cases across 23 methods; discovery also reports 213.
Full nonincremental solution build: zero warnings and
errors. Both sample assemblies ran successfully with each of the three models.

```powershell
dotnet build .\OnnxEmbeddings.slnx --no-incremental
dotnet test .\OnnxEmbeddings.slnx --logger "trx;LogFileName=shaped-tensor-followup-final.trx"
```

The runner is SDK-style **xUnit v2 on VSTest**, with .NET SDK 10.0.401. For a
unit-only run use `--filter "Category!=RealModels"` on the test project. For
all real-model cases use `--filter "Category=RealModels"`. Selected real tests
fail with an explicit setup error if assets/goldens are missing; none are skipped.

| Model | Independent rows | Python export vs PyTorch max abs | .NET public stages / provider max abs | Minimum .NET cosine | Maximum norm error |
|---|---:|---:|---:|---:|---:|
| MiniLM | 36 | 2.682209e-7 | 2.607703e-7 / 2.607703e-7 | 0.9999999999988272 | 6.050830e-8 |
| E5 | 67 | 2.682209e-7 | 2.682209e-7 / 2.682209e-7 | 0.9999999999981260 | 7.451196e-8 |
| Granite | 36 | 5.532057e-7 | 5.541370e-7 / 5.541370e-7 | 0.9999999999959595 | 9.839757e-8 |

IDs, masks and token types match **exactly** for all 139 rows. All models produce
384-dimensional vectors. Observed same-input-alone versus mixed-batch maximum
error is **0** for all three on this CPU, including Granite's nonzero pad ID.
Three synchronized concurrent callers per model pass. Providers use native
batches of at most two in parity tests versus six in the Python reference,
checking chunking, padding, count and order rather than merely reusing a matching
batch shape. E5 also passes 35 Query and 32 Document rows through the MEAI
single-purpose interface and mixed-role calls under both configured defaults.

Fresh observations are under `shapedTensorValidation` in
`evidence\dotnet-validation.json`, including exact commands, retained-case counts,
new test inventory and per-model results. Earlier top-level/model and
`tokenizerRefactorValidation` fields are historical, not substituted for this gate.

## Reference generation

After the explicit acquisition step in the README:

```powershell
.\.venv\Scripts\python.exe .\dev\reference.py
.\.venv\Scripts\python.exe .\dev\tokenizer_reference.py
```

The maintained, version-pinned tools in `dev\requirements.txt` (fully resolved
in `dev\requirements.lock.txt`) run only inside the checkout's `.venv`. They load local safetensors with HF AutoModel in Float32,
eager attention, eval mode, no gradients and two CPU threads. AutoTokenizer
generates independent exact IDs/masks at recipe limits with right padding.
Pooling follows the pinned publisher recipe and PyTorch L2 normalization.

The checked corpus has 36 MiniLM cases, 67 E5 cases and 36 Granite cases:
empty text; spaces, tabs/newlines and leading/trailing whitespace; mixed case,
diacritics, combining marks, CJK, emoji/ZWJ sequences, supplementary letters and
numbers; Unicode whitespace, punctuation, control/replacement characters;
literal special-token strings (including Granite `<mask>` left-whitespace
consumption); caller text already resembling E5 prefixes; short retrieval
examples; and lengths immediately below/at/above the truncation boundary,
including a multi-token emoji at that boundary. E5 tests both query and document
purpose for every ordinary case. Batches contain different lengths.
The extended regression cases also cover in-word tabs/newlines, format/private-use
characters, lowercase special-token lookalikes, dotted-I/Angstrom/Greek casing,
Hangul decomposition, and BMP/supplementary Chinese range boundaries.

The exact graph inputs and outputs, source revisions, Python tool versions and
observed export errors are in `evidence\python-validation.json`; downloaded file
hashes and exact pinned URLs are in `evidence\asset-provenance.json`. Model and
tokenizer assets themselves are not committed. Goldens are regenerated in the
ignored model directories, not substituted with .NET-generated expectations.

The Float32 final-vector gate is **maximum componentwise absolute error <=
2e-5**, cosine >= 0.99999 and norm error <= 2e-6. Float32 attention/GEMM
reassociation across PyTorch and ORT and vectorized pooling can vary by hardware;
the tolerance is deliberately above observed CPU errors, but tight enough to
catch different pooling, truncation, prefixes or weights. It is not a guarantee
for quantized or other unverified backends.

Contract tests use tiny **synthetic, untrained** ONNX fixtures, reproducibly
generated by `dev\make_contract_graphs.py`. They contain no publisher weights or
tokenizer assets. A 34-entry, hand-authored test vocabulary with deliberate
negative controls supports BERT regression tests; it is not a downloaded model
vocabulary. Unit/contract tests do not require Python after checkout.
The separate tokenizer-contract generator uses the same pinned offline HF
libraries. It adds 16 independent ID/token/decode rows with tiny synthetic
vocabularies (28 BERT entries, 268 byte-BPE entries and seven merges), plus
21 tokenizer-only rows using the three local publisher vocabularies. These
do not overwrite the original 139 model-reference rows. Full publisher Granite
decode matches HF for the valid sequences; BERT cleanup differences are explicit
below. Synthetic files are committed; publisher assets/references remain ignored.
The only ownership reflection is a test-only probe of pinned ORT's disposed flag
to avoid invoking an externally disposed native session; public stages, samples,
and providers do not use reflection.

Deterministic cancellation coverage checks pre-cancelled calls, enumeration,
and boundaries between texts/batches. The production scorer wires per-call
native termination, but exact in-flight cancellation latency, native OOM, and
non-CPU execution providers are not measured/tested. Concurrent disposal or
using a borrowed session after its owner disposes it is outside the supported
lifetime contract.

## Reproduce the built-in RoBERTa comparison

```powershell
dotnet run --project .\tools\Experiments -- roberta .\.assets\granite
```

The reproduction uses the **pinned Granite vocabulary and merges**. It supplies
an identity Fairseq mapping for IDs >=4, neutralizing the independent vocabulary
remapping issue before testing byte preprocessing. Content-only results:

| Input | HF expected | `EnglishRobertaTokenizer` direct | Same built-in, UTF8 bytes represented as Latin1 | Local BPE adapter |
|---|---|---|---|---|
| `hello` | `[42891]` | `[42891]` | `[42891]` | `[42891]` |
| `é` | `[1140]` | `[41907]` | `[1140]` | `[1140]` |
| `😀` | `[18636,7471]` | `[]` | `[18636,7471]` | `[18636,7471]` |
| literal `<mask>` | `[50264]` | `[41552,43776,15698]` | `[41552,43776,15698]` | `[50264]` |

This does **not** prove that the built-in BPE engine is unusable. It proves that
the built-in is not a drop-in Granite tokenizer, and that the observed Unicode
problem is byte preprocessing rather than just ID remapping. A UTF8 adapter
fixes the shown byte cases but not added-token semantics; correct word
pre-segmentation is another distinct responsibility. The implementation instead
uses the public `BpeTokenizer` with direct HF vocabulary IDs,
**`BpeOptions.ByteLevel = true`**, configured `SpecialTokens`, and a standard
custom `PreTokenizer` for GPT-2-style boundaries, added tokens and mask
left-whitespace consumption. The repro also checks this configured byte-level
engine directly against the HF IDs; no missing byte-BPE capability is claimed.
BOS/EOS, truncation and padding stay in the shared
preparer. No model engine was copied or handwritten. Passing this short repro
alone is not the full tokenizer verification; the complete golden corpus is
checked independently by the real-model suite.

## BERT failures found and fixed

```powershell
dotnet run --project .\tools\Experiments -- bert .\.assets\minilm
```

The initial implementation failed real-model parity even though ordinary
retrieval samples worked. The earliest discrepancy was the emoji input:
HF content IDs `[100,100]` became `[]`, producing final-vector errors as large
as 0.292 (MiniLM), not Float32 rounding noise. Expectations were **not changed**.

The pinned `BertTokenizer` uses `CreateWordOrPunctuation` with `\w+|[\p{P}]`
by default. That drops astral text and non-punctuation symbols instead of
passing them to WordPiece as unknown words or ASCII punctuation. Its normalizer
also removes tab/newline separators and recomposes accent-stripped text.
Changing `SplitOnSpecialTokens` to true fixes uppercase added tokens but also
recognizes lowercase lookalikes after normalization, unlike the pinned HF
`normalized=false` added tokens. The reproduction deliberately uses true for
that flag and accent stripping, not a deliberately misconfigured baseline.

| Input | HF content IDs | Configured built-in | Adapter |
|---|---|---|---|
| `😀🚀 👩‍💻` | `[100,100]` | `[]` | `[100,100]` |
| `a\tb\nc\rd` | `[1037,1038,1039,1040]` | `[5925,2094]` | HF exact |
| `[cls] [mask] [CLS] [MASK]` | `[1031,18856,2015,1033,1031,7308,1033,101,103]` | `[101,103,101,103]` | HF exact |
| `한국어` | `[1469,30006,30021,29991,30014,30020,29999,30008]` | `[100]` | HF exact |

`BertUncasedTokenizer` configures Microsoft's public `Normalizer` and
`PreTokenizer` extension points. Its normalizer protects exact uppercase added
tokens before normalizing ordinary fragments with the pinned HF
clean-text/Chinese/NFD/accent/lowercase policy. Its pretokenizer segments those
specials, Unicode whitespace and ASCII/Unicode punctuation without discarding
symbols. This expresses the required ordering without replacing the engine or
requiring a Tokenizers change. It delegates **all WordPiece segmentation and
vocabulary lookup to Microsoft's BertTokenizer**. The inner convenience
`EncodeToIds` overload that automatically inserts framing is never used.
Malformed UTF-16 is rejected explicitly. The expanded independent corpus
preserves the original failing cases and adds focused regression cases.
Full repro output is in `evidence\bert-repro.jsonl`.

## Standard Tokenizer contract

The former public callable wrapper has been removed. The `PrepareBatch` extension
accepts `Microsoft.ML.Tokenizers.Tokenizer` plus immutable
`TokenSequenceOptions(maximumSequenceLength, beginningTokenId, endTokenId, paddingTokenId)`.
The two public adapters are `BertUncasedTokenizer` and
`Granite30MEnglishTokenizer`; neither hides standard methods with `new` or
implements an encode-only substitute. All named providers retain their
constructors. The shaped-stage milestone does not change these tokenizer engines
or their encoding/decoding policies.

| Surface | Contract |
|---|---|
| String/span encode, count | Same content IDs and values through concrete and base variables. No implicit framing. Malformed UTF-16 rejected. Null string follows the base API's empty-span behavior. |
| Normalizer/pretokenizer | Exposed standard objects. BERT normalization can be disabled; Granite has no normalizer. Disabling pretokenization disables ordinary and added-token splitting. Default flags are required for model parity. |
| Coordinates | UTF-16 in BERT `NormalizedText` when enabled, original input otherwise. Granite uses original text and null `NormalizedText`. Offsets include full scalars and may overlap for byte tokens; no byte-alphabet index masquerades as a text offset. |
| Added tokens | Exact BERT uppercase tokens survive ordinary lowercasing. Granite `<mask>` consumes preceding whitespace; its offset covers the literal token, excluding discarded whitespace (not HF's lstrip-inclusive offset convention). |
| Bounded encode/index | Positive budgets only. Overlapping source ranges stay together; fewer than the budget may fit. Forward consumption/index is the last retained end, or zero. Reverse index is the first retained start, or text length. All-fit includes the entire processed text, including ignored leading/trailing whitespace (forward length, reverse zero). |
| Boundary meaning | Selects token prefixes/suffixes from full tokenization, not a promise that independently re-tokenizing a substring preserves WordPiece continuation IDs. Normalization is not an original-to-normalized alignment map. |
| Batch truncation | Full content IDs are encoded and then truncated, so a byte-token group can be cut exactly as in HF model inputs. This is deliberately different from a standalone UTF-16 boundary query. |
| BERT decode | Both string/span use Microsoft's WordPiece span spacing, retaining special-token strings; no original-casing reconstruction or HF punctuation cleanup promise. |
| Granite decode | Inverse byte alphabet plus BCL UTF-8 validation across consecutive token IDs. Complete groups are atomic in small destinations. Invalid IDs/invalid or incomplete UTF-8 produce `InvalidData`; string overload throws `InvalidOperationException`. |
| Decode ownership | Caller owns destination. Counts describe a valid written prefix; no partial token group or surrogate is consumed. This is not a resumable decoder with retained WordPiece spacing state. String decode snapshots the enumerable once before any buffer growth. Null enumerables throw. |

An actual `Tokenizer` supplied by Microsoft can also drive preparation; no
adapter type check is used. The caller must provide a content-only vocabulary
contract and matching sequence options. In particular, the Microsoft concrete
`BertTokenizer` convenience methods insert specials by default whereas dispatch
through its base `Tokenizer` differs. Our adapters do not expose that ambiguity.

Two pinned BPE contract discrepancies and a WordPiece overload distinction
justify the remaining narrow wrapper logic:

```powershell
dotnet run --project .\tools\Experiments -- contract .\.assets
```

With the exact Granite vocabulary and `BpeOptions.ByteLevel=true`,
`😀` has correct IDs `[18636,7471]`, but the pinned engine reports both
UTF-16 offsets as `[0,1)` (half a surrogate pair). The adapter reports `[0,2)`
for both. Decoding only `[18636]` with a 32-character destination returns
`Done, idsConsumed=0, charsWritten=0` in the pinned span decoder; the adapter
returns `InvalidData`. A complete pair with a one-character destination returns
`DestinationTooSmall, 0, 0`, and with two characters returns `Done, 2, 2`.
The explicit strict decoder also avoids the pinned byte decoder's fixed-size
intermediate token buffers. Strict rejection of incomplete bytes differs from
HF's replacement-character decoding policy; it does not change model input IDs.

For BERT IDs `[7592,1012]`, Microsoft's string decoder yields `hello.` while
its span decoder yields `hello .`; both adapter overloads deliberately yield
`hello .`. These are measured scoped behaviors, not claims about every release.
Full output is recorded in `evidence\tokenizer-contract-repro.jsonl`.
Neither the byte offset fix nor decoding introduces a new BPE/WordPiece engine.

## Tensor interop experiment

```powershell
dotnet run -c Release --project .\tools\Experiments -- interop .\tests\CommunityToolkit.Embeddings.Onnx.Tests\Fixtures
```

This bounded experiment uses a tiny synthetic, untrained graph and the exact
consumer combination: **net10.0, ORT/ORT.Managed 1.23.2, Tensors 10.0.9**.
Unsafe code and the single `SYSLIB5001` suppression live only in the experiment,
not the production library. Its JSONL is recorded in `evidence\tensor-interop.jsonl`.
These three findings are independent:

1. **Binary incompatibility.** `CreateTensorValueFromSystemNumericsTensorObject`
   compiles, then throws `MissingMethodException` for
   `Tensor.Create(ReadOnlySpan<IntPtr>, Boolean)`. Dense, offset, strided,
   singleton-axis, rank-zero empty and shaped-empty inputs all hit this binding
   failure before layout-specific behavior can be established. The source's
   conditional noncontiguous copy and reflection-based pinning are **source
   observations**, not successfully exercised copy/lifetime paths with these
   binaries. There is no measured bridge throughput and no silent fallback.
2. **Tensor slice pinning offset.** For backing `[91,92,11,12,21,22]`, shape
   `[3,2]`, slice start `[1,0]`, logical values are `[11,12,21,22]`.
   `GetPinnedHandle()` instead points at `91`; native ONNX sees
   **`[91,92,11,12]`** when given the slice's `[2,2]` shape.
   The handle remains alive throughout every pointer/native access.
3. **Stable control.** `CreateTensorValueFromMemory` with
   `data.AsMemory(2,4)` yields **`[11,12,21,22]`** in the same native graph.
   Changing the source's first sliced value to `111` remains visible after a
   compacting GC, proving an alias rather than a hidden copy in this control.

The public pin primitive **already exists**. The proposed follow-up is not to
add another pin API: first fix/verify offset semantics, then validate layout and
ownership in consumers. A lexical pin of a validated, contiguous **tensor span**
also gives the expected offset values after compacting GC, but cannot escape
its `fixed` scope. It is an experiment, not a new production ownership wrapper.
The strided probe is rejected explicitly; separately copying its logical values
produces `[91,92,12,21]` and is isolated from subsequent source mutation.
The pinned permutation and broadcast probes both report `IsDense=false` and
reject a whole logical `GetSpan`; core pooling still checks actual strides and
offset-aware access instead of relying on a flag as its complete contract.

Stable binding accepts `[0,0]` empty input. `Tensor.CreateFromShape<float>([0,0,3])`
preserves rank three and zero elements. Oversized shape and wrong graph element
type are rejected explicitly. A tracked `MemoryManager` verifies one pin/one
unpin on normal, injected-exception, and deterministic cancellation exits;
invalid shape is rejected before pinning (zero/zero). Its owner survives GC while
the native value is live. These checks do not dereference freed pointers and do
not establish exact native cancellation latency or bridge cleanup after its
unreachable native-construction paths.

Existing upstream work is credited:
[microsoft/onnxruntime#25460](https://github.com/microsoft/onnxruntime/issues/25460)
reported older ORT/Tensors-preview compatibility and singleton-layout problems;
its missing `CopyTo` signature differs from this repro's missing `Create`.
[microsoft/onnxruntime#25972](https://github.com/microsoft/onnxruntime/pull/25972)
already proposes Tensors 10 and reflection-free `GetPinnedHandle()` integration.
This prototype did **not** build that PR and does not claim it fixes the separately
reproduced slice-offset behavior. The independently isolated pinning report is
[dotnet/runtime#134691](https://github.com/dotnet/runtime/issues/134691).
Minimal, model-free .NET 10 file apps are published separately for
[tensor pinning](https://gist.github.com/luisquintanilla/b231097fc18b5564d9e5a0f7e826e2ae)
and [the ORT binary failure](https://gist.github.com/luisquintanilla/eeffb849e6f3a350304d37e346b6f606).
No upstream implementation or package update is included here.

Production keeps the stable owned-memory input path. Shaped output is still
copied once out of native memory. Pooling and caller conversion use public
offset-aware tensor spans and explicit row-major layout checks, never the
defective tensor pin path or reflection.

An additional shape regression found that a `[1,2,1]` view over `[2,6]`
cannot supply its full two-element `GetSpan` in the pinned package, despite
contiguous logical storage. The `singleton-span-control` probe records the
direct `TryGetSpan` failure and successful public `Squeeze` control. After explicit
layout validation, pooling removes singleton axes from view metadata before
accessing the contiguous span; a one-element view uses the safe single-reference
`Span` constructor from its logical indexer. No data is materialized or pinned.
Cross-type overlap is also rejected: `MemoryMarshal.Cast<long,float>` can safely
construct a destination aliasing attention bytes, so checking only float hidden
states would not protect the input mask.

### Trimmed and NativeAOT scope

The existing experiment project has a small `InteropSmoke` compile selection,
not another library/project. It excludes JSON/reflection diagnostics and runs
only a stable sliced-memory input and owned shaped output through the synthetic
graph:

```powershell
dotnet publish .\tools\Experiments -c Release -r win-x64 -p:InteropSmoke=true -p:PublishTrimmed=true -p:NuGetLockFilePath=obj\interop-smoke.packages.lock.json -o .\.assets\interop-trimmed
.\.assets\interop-trimmed\Experiments.exe .\tests\CommunityToolkit.Embeddings.Onnx.Tests\Fixtures
```

This trimmed, self-contained smoke **published without warnings and ran successfully**:
`PASS: shaped offset input, stable Memory ORT binding, owned shaped output [2,2,3].`
It is not a claim that every provider/tokenizer or the incompatible bridge is
trim/AOT validated. Smoke-specific restore artifacts stay ignored; normal package
versions and five project lockfiles are unchanged.

The corresponding `-p:PublishAot=true` publish was attempted after restoring its
compiler assets. Native compilation was **blocked by the unavailable/discoverable
Windows C++ platform linker**. No toolchain was installed, no NativeAOT executable
ran, and a successful managed/trimmed publish is not counted as NativeAOT success.
Restore the normal graph afterwards with `dotnet restore .\OnnxEmbeddings.slnx --locked-mode`.

Binding-only measurements (32 x 256 Int64, 100 warmups, median seven alternating
rounds of 1,000 construct/dispose calls) observed **0.4879 us / 72 managed bytes**
for stable Memory binding versus **0.4619 us / 72 bytes** for the constrained
lexical-span pin. Shape/tensor construction and inference are excluded. This tiny
timing difference is not a reliable speedup claim, and the lexical lifetime is
not a drop-in persistent binding. The shipped experimental bridge has no valid
comparison number because it fails binary binding.

## Focused measurements

```powershell
dotnet build -c Release .\tools\Experiments
$env:DOTNET_TieredCompilation = "0"
dotnet run -c Release --project .\tools\Experiments --no-build -- measure .\.assets > .\.assets\measurements.json
Remove-Item Env:DOTNET_TieredCompilation
```

Run after correctness checks and without competing tests or builds. The harness
first verifies equivalence of baseline and helper outputs. It compares complete
preparation using the **same tokenizer** against simple LINQ/list allocation and
copying; then separately compares scalar masked mean (MiniLM) or CLS (Granite)
plus safe normalization with the shared `TensorPrimitives` implementation. Both pooling paths validate
nonfinite outputs. Batch sizes are 1, 8 and 32, padded sequence lengths 16, 128
and the recipe maximum (256 for MiniLM, 512 for Granite), with mixed lengths.
Preparation includes token types for MiniLM and omits them for Granite, matching
the inspected graphs. Their complete input tensors agree. Scalar pooling uses
the old jagged result representation; current pooling returns a rank-two tensor.
All numerical elements are compared before measurement.

The current **`shaped-tensor-v1`** harness adds two separate storage comparisons at
the recipe limit for batches 1, 8 and 32: allocating `Pool` versus reusable
`PoolInto`, and `Pool` plus safe row copies into MEAI versus direct pooling into
the provider's private final buffer with row memories. The latter uses the same
pooling arithmetic on both sides; it isolates the avoidable tensor-to-MEAI copy
chain rather than conflating it with scalar/SIMD arithmetic. The encoder/native
output copy is unchanged and not included in these storage comparisons.
`pool-meai-jagged-storage` separately models the previous jagged-array ownership
policy with scalar arithmetic and direct MEAI row wrapping. Its **allocation**
comparison does not pretend that scalar versus SIMD timing isolates storage.

An additional representative **prepare/score/pool/MEAI** comparison uses eight
mixed-length inputs padded to 128 tokens for MiniLM and Granite, two CPU ORT threads,
and three calls per round (seven alternating rounds, ten warmups).
It compares full stages with safe pooled-row copies against the actual provider's
private-buffer path. Both include tokenization, native inference, the same single
native-output copy and final result objects. Native allocations are still not
counted. Inference dominates this comparison; close timings are not evidence
of a throughput improvement.

Steady-state values are medians of seven alternating-order rounds of 30 calls
after ten warmups, measured with `Stopwatch` and
`GC.GetAllocatedBytesForCurrentThread`. Output allocation is included.
Tiered compilation is disabled for the measurement process to avoid JIT tier
transitions inside the short rounds; this is recorded in the result.
Caller-input creation, tokenizer construction and native allocations are excluded.
Native inference is included only in the explicitly end-to-end comparison.
Repeated text intentionally exercises tokenizer
caches; these are not cold-cache throughput or end-to-end serving benchmarks.
First measured calls are separately reported and are not characterized as a
process-cold benchmark. Tokenizer construction and the first preparation of an
uncached sentence are additionally reported separately under `startup`; runtime
startup itself is excluded. Byte counts are **managed thread allocations**, not
working set, peak memory, or "zero allocations".

Results are recorded in `evidence\measurements.json`. They are local
observations, not a speedup guarantee for another machine or a hypothetical
upstream API.

Representative **current `shaped-tensor-v1`** results (Windows x64, .NET 10.0.12,
16 logical processors; us/call and managed bytes/call):

| Work / shape | Baseline us | Current us | Baseline bytes | Current bytes |
|---|---:|---:|---:|---:|
| MiniLM preparation, 32 x 256 | 2,200.76 | 1,912.03 | 1,738,808 | 1,460,624 |
| Mean + normalize, 32 x 256 x 384 | 4,171.80 | 2,035.44 | 50,200 | 49,320 |
| Granite preparation, 32 x 512 | 7,220.82 | 6,493.45 | 6,051,584 | 5,642,352 |
| CLS + normalize, 32 x 512 x 384 | 3,954.76 | 3,789.07 | 50,200 | 49,320 |
| Tensor pool + MEAI row copies vs private MEAI buffer, mean | 2,074.57 | 2,206.43 | 102,304 | 52,384 |
| Tensor pool + MEAI row copies vs private MEAI buffer, CLS | 4,035.22 | 4,045.54 | 102,304 | 52,384 |
| Old-style jagged MEAI storage vs private buffer, mean | 3,872.46 | 1,890.66 | 53,408 | 52,384 |
| Allocating pool vs reused destination, mean | 1,926.47 | 1,983.56 | 49,320 | 0 |
| MiniLM full pipeline, 8 x 128 | 163,012.27 | 168,918.17 | 1,784,536 | 1,772,472 |
| Granite full pipeline, 8 x 128 | 165,015.63 | 167,061.13 | 1,952,672 | 1,940,608 |

The private result buffer saves **49,920 bytes** against the avoidable
tensor-plus-row-copy path, but only **1,024 bytes** against old-style jagged MEAI
storage for 32 x 384 results. Shaped allocation-returning pooling saves 880 bytes
versus its jagged scalar baseline. Preparation allocations are unchanged from
`standard-tokenizer-v1`; adoption of tensor views does not remove token records
or widening buffers. `PoolInto` shows zero *steady-state current-thread managed*
allocation for this supplied/reused destination, not zero-allocation inference.

There is **no end-to-end speedup claim**: the current provider measured about
3.6% slower for MiniLM and 1.2% slower for Granite than the alternative composed
copying path in this run, despite lower managed allocations. Native inference
dominates and short timing samples fluctuate. Both paths retain the native-output
copy; a tensor object adds shape metadata rather than eliminating that allocation.
Cross-implementation wall-clock comparisons against historical runs below are
not controlled experiments.

Historical **published `standard-tokenizer-v1` (commit `9623ac5`)** measurements on Windows
x64, .NET 10.0.12, 16 logical
processors (time is microseconds per call; bytes are allocated per call):

| Work / shape | Baseline us | Helper us | Baseline bytes | Helper bytes |
|---|---:|---:|---:|---:|
| MiniLM preparation, 1 x 256 | 54.08 | 50.53 | 54,984 | 45,872 |
| MiniLM preparation, 32 x 256 | 1,924.99 | 1,761.68 | 1,738,808 | 1,460,624 |
| MiniLM mean + normalize, 32 x 256 x 384 | 3,354.10 | 1,775.55 | 50,200 | 50,200 |
| Granite preparation, 1 x 512 | 205.76 | 199.19 | 189,864 | 176,680 |
| Granite preparation, 32 x 512 | 6,880.88 | 5,969.73 | 6,051,668 | 5,642,352 |
| Granite CLS + normalize, 32 x 512 x 384 | 3,010.28 | 2,964.13 | 50,200 | 50,200 |

In that implementation the helper reduced preparation allocations versus its baseline, but still
allocates substantially. The small CLS timing difference is **not evidence of
a reliable speedup**; scanning token states for nonfinite values dominates its
work. No blanket acceleration claim follows. Tokenizer construction
cost 30.64 ms / 6,212,224 managed bytes for MiniLM and 133.52 ms / 34,194,024 bytes
for Granite; first uncached preparation cost 19.06 ms and 35.59 ms respectively.
These are startup observations, not steady-state timings or model-load timings.

The old measurements are preserved only in published commit `f6d174f`, not
presented as current. Relative to that implementation, the 32 x 256 MiniLM
helper allocation increased from 1,150,544 to 1,460,624 bytes: standard token
records and offset-aware contract handling now materialize before ID extraction.
Granite 32 x 512 helper allocation decreased from 7,059,968 to 5,642,352 bytes
after delegating forward byte handling to `BpeOptions.ByteLevel`. Granite
construction now also retains an inverse vocabulary for validated decoding.
These cross-implementation allocation observations do not isolate throughput
effects; timing runs at different times are not a controlled refactor speedup
experiment. No bounded-memory, zero-allocation, or zero-copy claim is made.

## Dependency boundary

[`evidence/dependencies.json`](evidence/dependencies.json) inventories the actual
restored transitive graphs for all five projects. The library contains exactly
the four direct packages in the README and **Google.Protobuf 3.30.2** transitively
through Tokenizers. It contains **no native ORT package**. The runnable projects
explicitly add matching CPU ORT; tests add only the test SDK/adapter/framework
and their dependencies. There is no ML.NET pipeline, Semantic Kernel,
Agent Framework or vector-store dependency. Lock files preserve resolved
versions and NuGet content hashes; `dotnet restore --locked-mode` succeeds.

`dev\record_evidence.py` checks acceptance flags before recording the JSON
evidence and rejects forbidden framework packages in resolved graphs. The asset
acquisition script also rechecked every pinned artifact hash against the recorded
evidence. This is developer validation, not a mandatory consumer asset allowlist.

## Requirement-to-evidence map

All named tests are in `tests\CommunityToolkit.Embeddings.Onnx.Tests`. This is a
behavioral checklist, not a measured line-coverage or empirical mutation score.

| Requirement (request wording) | Evidence |
|---|---|
| "all three concrete providers" | `PinnedModels_MatchIndependentTokensAndVectors` for MiniLM, E5 and Granite |
| "SAME public underlying components" | That test compares both direct stages and providers; independent `samples\Composition` calls public prepare/score/pool |
| "Token IDs/masks exact; numerical tolerances justified and report observed max errors." | `PinnedModels_MatchIndependentTokensAndVectors`; all 139 rows; table and `evidence\dotnet-validation.json` |
| "Compare final vectors as well as simple ranking; ranking alone is not parity." | Above per-component comparisons plus `PinnedModels_ConcreteValidationOwnershipAndRetrievalRanking` |
| "empty/whitespace text, Unicode/diacritics/punctuation, leading spaces, truncation boundaries, mixed lengths, nonzero pad id (Granite), and E5 roles" | All-model independent corpus; `Prepare_BudgetsSpecialsTruncatesAndPadsInOrder`; `Prepare_LengthTwoKeepsOnlyBosEos` |
| "Ensure a sentence yields equivalent result alone and in mixed length batches." | `PinnedModels_SingleAndConcurrentInferenceMatchMixedBatches`, every reference row; observed max error 0 |
| "single/mixed batch semantics" | `E5_DefaultSingleRoleViaMeaiAndTypedMixedRolesMatchGoldens`; `Format_PreservesCallerTextAndAlwaysAddsSelectedPrefix` |
| "Empty input sequence is a legitimate empty result; null/invalid inputs are explicit errors." | `Prepare_EmptyBatchAndEmptyTextAreDistinct`; `Prepare_RejectsInvalidArgumentsAndTokenizerResults`; `Generate_ValidatesFixedOptionsAndExposesMetadata` |
| "required names/types/shapes and compatible output contracts" | `Constructor_RejectsInvalidGraphContract`; `Score_RejectsStaticBatchAndSequenceShapesAndRuntimeOutputDimensions`; Granite's pooled output is explicitly rejected |
| "Inspect supported optional token_type_ids" | `Score_OptionalTypesShapesCancellationAndManagedOwnership` exercises actual graphs with/without the input |
| "all-padding/zero/nonfinite policy" | `Pool_RejectsPaddingZeroAndNonfiniteEvenWhenMasked`; `Normalize_RejectsZeroEmptyAndNonfinite`; `Normalize_ExtremeMagnitudesAreStable` |
| "resource ownership" | `Constructor_CopiesAndValidatesBuffers`; `Dispose_OwnedAndBorrowedSessionsBehaveDifferently`; `Load_BorrowsOptionsAndValidatesPaths`; `Dispose_OwnedAndBorrowedEncodersAndServices` |
| "Cancellation and disposal/error cleanup must be coherent" | `Prepare_CancellationStopsBeforeAndBetweenRows`; `Generate_CancellationIsObservedThroughEnumerationAndBetweenBatches`; `Score_OptionalTypesShapesCancellationAndManagedOwnership`; native timing limitation above |
| "Respect native session concurrency supported by configuration" | `PinnedModels_SingleAndConcurrentInferenceMatchMixedBatches`, three concurrent CPU callers per model |
| "allocations and time for representative batch sizes/lengths (use simple baseline vs helper)" | `tools\Experiments\Measurements.cs`; `evidence\measurements.json`; baseline/helper equality verified before measuring |
| "package/transitive dependency inventory proving no ML.NET/SK/AF" | `evidence\dependencies.json`, all five actual restored graphs |
| "runnable provider and composition samples" | All six documented model/sample combinations executed successfully |
| "DEFER ML.NET integration" | No ML.NET project or adapter; single focused library plus two examples, experiments and tests |

### Shaped-tensor milestone requirements

| Requirement | Evidence |
|---|---|
| One standard tokenizer extension, no replacement preparer hierarchy | `PrepareBatch_DefaultLimitIs32AndOldPublicPreparerIsRemoved`; ordinary WordPiece/BPE preparation regressions; `Generator_ExposesDirectCompositionWithoutPreparer` |
| Read-only `[B,S]` views, copied caller inputs, explicit optional input | `ReadOnlyViews_HaveShapeAndCopiedStorageWithoutMutablePublicAliases`; `EmptyViews_PreserveTwoAxesAndDistinguishAbsentTokenTypes` |
| Owned `[B,S,H]` result; rank-preserving empty state | `Score_OptionalTypesShapesCancellationAndManagedOwnership`; `Score_EmptyTensorPreservesShapeUnlikePublicTensorEmpty` |
| `[B,H]` pooled result and reusable destination | `PoolInto_OverwritesAndReusesDestination`; `Pool_EmptyShapeIsPreservedAndPoolIntoDoesNotTouchBackingStorage` |
| Rank, dimensions, checked products, nontrivial layout rejection | `Pool_RejectsInvalidRanksShapesAndDestinationDimensions`; `Pool_RejectsDimensionAndProductOverflowBeforeMaterializingBuffers`; nine `Pool_RejectsNonRowMajorLayoutsIncludingPermutations` cases |
| Nonzero offsets and singleton axes preserve logical values | `Pool_UsesLogicalNonzeroOffsetSlicesNotArrayPrefix`; `PoolInto_AcceptsSingletonHiddenDimension`; `PoolInto_SingleElementAndSqueezedOffsetViewsUseLogicalValues`; `PoolInto_OffsetMultirowSequenceWithSingletonHiddenUsesLogicalValuesAndPreservesGuards` asserts means `[4,10]` and unchanged sentinels/inputs |
| Destination cannot overwrite either input | `PoolInto_RejectsOverlappingStorage`; `PoolInto_RejectsDestinationAliasingAttentionMaskBytes`; `PoolInto_AllowsDisjointViewsOfSameArray` |
| Mask/finite/zero/all-padding/CLS/overflow policies | Existing regressions plus `Pool_DirectAttentionViewRejectsNonBinaryValues`; `PoolInto_RejectsInvalidNumericsIncludingMaskedStates`; `PoolInto_RejectsAccumulationOverflowWithAndWithoutNormalization` |
| Final MEAI ownership, row order, later calls and disposal | `Generate_RowSlicesRemainOwnedAcrossCallsAndDisposal` checks private per-batch buffers, disjoint row ranges, numerical values, setter isolation and retained memory; composition sample copies arbitrary mutable tensor rows |
| Exact independent model behavior retained | All 139 original rows; same tolerances; all 157 prior test cases retained; fresh all-model and full-suite evidence |
| Experimental bridge assessed rather than silently adopted | `TensorInteropProbe.cs`, source attribution, recorded binary failure, native offset controls, explicit copy, GC and balanced-pin observations |
| Measured costs, no automatic zero-copy claim | Current `shaped-tensor-v1` evidence: preparation, scalar/shaped pooling, old jagged ownership, copy chain, reusable destination, representative end-to-end paths and binding-only comparison |

Final static test-gap and assertion-quality reviews found no remaining in-scope
findings. This is not an empirical mutation run or line-coverage claim. The
test inventory in `shapedTensorValidation.newTestMethods` records exact names
and case counts.

### Standard-tokenizer refactor requirements

| Requirement (request wording) | Evidence |
|---|---|
| "use Microsoft.ML.Tokenizers.Tokenizer rather than the custom public TextTokenizer wrapper" | `Adapters_AreSealedStandardTokenizersWithoutHiddenEncodingOverloads`; removed wrapper and former internal adapters; two public sealed standard tokenizers |
| "consume the STANDARD Tokenizer plus explicit immutable sequence/model-input configuration" | `Prepare_AcceptsOrdinaryWordPieceTokenizerAndExposesSequenceOptions`; `SequenceOptions_ValidBoundariesAreImmutable`; `SequenceOptions_RejectsLengthsBelowTwo`; now exposed as `Tokenizer.PrepareBatch` |
| "ONE owner for surrounding special tokens" | `Bert_SpecialTokensAreExtractedBeforeOrdinaryLowercasing`; `Granite_ConcreteAndBaseCallsLeaveSurroundingTokensToPreparation`, both base/concrete string/span dispatch |
| "coherent string/span encoding, EncodeToTokens with valid token values/offsets" | `Adapters_IdsTokenValuesCountAndDecodeMatchIndependentSyntheticHf`; `Bert_NormalizedOffsetsAccountForAccentRemovalWhitespaceAndPreservedSpecials`; `Granite_OriginalUtf16OffsetsCoverWholeScalarsAndByteTokensMayOverlap` |
| "bounded encoding and charsConsumed/NormalizedText, CountTokens, token-boundary queries in both directions" | `Bert_BoundedWordPiecesAndReverseIndexFollowAdvertisedOffsets`; `Granite_BoundedEncodingAndIndicesNeverSplitOverlappingScalarTokens`; `Bert_WhitespaceOnlyFullConsumptionIncludesTrailingWhitespace`; `Adapters_NonPositiveBudgetsThrowForEmptyAndNonemptyStringAndSpan` |
| "Respect consideration flags consistently with documented behavior" | `Adapters_BoundedQueriesHonorDisabledPreTokenizationAndNormalization`; `Bert_PreTokenizationFalseDisablesAllSplittingButWholeSpecialRemainsKnown`; `Granite_PreTokenizationFalseDisablesAllSplittingButWholeSpecialRemainsKnown`; `Adapters_ExposeConfiguredNormalizerAndPreTokenizerObjectsWithConcreteBehavior` |
| "malformed UTF16 validation" | `Adapters_MalformedUtf16RejectedAcrossEveryEncodingAndQuerySurface`; preparation callback never receives invalid text in `EncodeContent_InvalidUtf16IsRejectedBeforeDelegateOrPreparation` |
| "follow upstream null-as-empty consistently" | `Adapters_NullEmptyAndDefaultSpanHaveTheSameEmptyContentContract`; batch rejects null in `Prepare_RejectsInvalidArgumentsAndTokenizerResults` |
| "string/span decoding" | `Bert_DecodeUsesWordPieceSpacingPreservesSpecialsAndCommitsWholeTokens`; `Granite_DecodePartialDestinationsCommitOnlyWholeUtf8Groups`; `Adapters_StringDecodeEnumeratesInputOnlyOnceIncludingBufferGrowth` |
| "Added tokens, multi-byte and supplementary characters, partial output spans, invalid IDs, and UTF8 byte sequences spanning tokens" | `Granite_DecodeMergedTokenAndAddedTokenRemainAtomic`; `Granite_DecodeTokensContainingPartialAndCompleteScalarsCommitAsOneGroup`; `Adapters_DecodeInvalidIdsReportFailureAndOnlyCommittedPrefix`; `Granite_StrictUtf8RejectsIncompleteOverlongSurrogateAndOrphanBytes` |
| "Preserve independently generated HF/PyTorch fixtures and all 139 existing input rows, same numeric gates" | `PinnedModels_MatchIndependentTokensAndVectors`; `PinnedModels_SingleAndConcurrentInferenceMatchMixedBatches`; `E5_DefaultSingleRoleViaMeaiAndTypedMixedRolesMatchGoldens`; exact row inventory and all previous cancellation/ownership tests retained |
| "Prefer independent HF reference evidence for semantic tokenization/decode" | `PinnedModels_StandardTokenizerMatchesIndependentUnicodeSpecialAndWhitespaceEvidence`; separate `dev\tokenizer_reference.py`, not .NET-generated expectations |
| "Rerun all six existing sample combinations" | Both `samples\Providers` and `samples\Composition` ran for MiniLM/E5/Granite in Release after production edits; current sequence API is used by composition, provider constructor calls remain unchanged |
| "rerun focused measurement harness before claiming current numbers" | `tools\Experiments\Measurements.cs`; current evidence labeled `shaped-tensor-v1`; earlier `standard-tokenizer-v1` numbers explicitly historical |
| "Keep dependencies unchanged" | Central versions and all five project lockfiles unchanged; only test fixture copy metadata added |

Focused regressions for the discovered BERT defects are
`Bert_EmojiAndAstralTextPreserveUnknownWordGroups`,
`Bert_AsciiAndUnicodeSymbolsAreNotDropped`,
`Bert_SpecialTokensAreExtractedBeforeOrdinaryLowercasing`,
`Bert_TabsAndNewlinesSeparateWordsRatherThanJoining`,
`Bert_LowercaseSpecialLookingTextRemainsOrdinary`, and
`Bert_HangulRemainsDecomposedForWordPiece`. Malformed/valid surrogate boundaries
are checked by `EncodeContent_InvalidUtf16IsRejectedBeforeDelegateOrPreparation`
and `EncodeContent_ValidSurrogatePairsReachDelegateUnchanged`.

## Primary evidence

- [MiniLM recipe limit](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/sentence_bert_config.json),
  [pooling](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/1_Pooling/config.json),
  and [tokenizer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer_config.json).
- [E5 model card and task prefixes](https://huggingface.co/intfloat/e5-small-v2/blob/ffb93f3bd4047442299a41ebb6fa998a38507c52/README.md).
- [Granite model card, CLS and normalization](https://huggingface.co/ibm-granite/granite-embedding-30m-english/blob/9b5b096411652ec1189c68fcfb90d0a82c5b45af/README.md)
  and [tokenizer configuration](https://huggingface.co/ibm-granite/granite-embedding-30m-english/blob/9b5b096411652ec1189c68fcfb90d0a82c5b45af/tokenizer_config.json).
- Microsoft [BERT construction](https://github.com/dotnet/machinelearning/blob/v5.0.0/src/Microsoft.ML.Tokenizers/Model/BertTokenizer.cs),
  [BERT normalization](https://github.com/dotnet/machinelearning/blob/v5.0.0/src/Microsoft.ML.Tokenizers/Normalizer/BertNormalizer.cs),
  and [EnglishRoBERTa](https://github.com/dotnet/machinelearning/blob/v5.0.0/src/Microsoft.ML.Tokenizers/Model/EnglishRobertaTokenizer.cs).
- Pinned standard [Tokenizer dispatch and contracts](https://github.com/dotnet/machinelearning/blob/v5.0.0/src/Microsoft.ML.Tokenizers/Tokenizer.cs),
  [BpeOptions.ByteLevel and extension points](https://github.com/dotnet/machinelearning/blob/v5.0.0/src/Microsoft.ML.Tokenizers/Model/BpeOptions.cs),
  [BPE encode/decode](https://github.com/dotnet/machinelearning/blob/v5.0.0/src/Microsoft.ML.Tokenizers/Model/BPETokenizer.cs),
  and [byte-to-source offsets](https://github.com/dotnet/machinelearning/blob/v5.0.0/src/Microsoft.ML.Tokenizers/Model/Word.cs).
- HF reference [BERT normalization](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/normalizers/bert.rs)
  and [punctuation policy](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/pre_tokenizers/bert.rs).
- [ORT 1.23.2 OrtValue ownership](https://github.com/microsoft/onnxruntime/blob/v1.23.2/csharp/src/Microsoft.ML.OnnxRuntime/OrtValue.shared.cs)
  and [MEAI provider dependency guidance](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai).
- Pinned [Tensor 10.0.9 views, slices and pinning](https://github.com/dotnet/runtime/blob/v10.0.9/src/libraries/System.Numerics.Tensors/src/System/Numerics/Tensors/netcore/Tensor_1.cs),
  [ReadOnlyTensorSpan offset-aware span access](https://github.com/dotnet/runtime/blob/v10.0.9/src/libraries/System.Numerics.Tensors/src/System/Numerics/Tensors/netcore/ReadOnlyTensorSpan_1.cs),
  and [MEAI 10.3.0 Embedding<T> memory ownership](https://github.com/dotnet/extensions/blob/v10.3.0/src/Libraries/Microsoft.Extensions.AI.Abstractions/Embeddings/Embedding%7BT%7D.cs).
