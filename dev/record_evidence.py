"""Copy generated validation evidence and inventory resolved .NET dependencies; never copy model assets."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "docs" / "evidence"
    destination.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((ROOT / ".assets" / "dotnet-validation.json").read_text())
    validation = evidence.get("shapedTensorValidation", {})
    tests = validation.get("tests", {})
    if (validation.get("passed") is not True or
        validation.get("baseline") != "9623ac5bbfc03e0614d71630c95e015df2e066fd" or
        validation.get("build", {}).get("exitCode") != 0 or
        tests.get("exitCode") != 0 or tests.get("failed") != 0 or tests.get("skipped") != 0 or
        tests.get("baselineCasesRetained") != 157 or tests.get("baselineCasesMissing") != 0 or
        validation.get("quality", {}).get("passed") is not True):
        raise RuntimeError("Fresh shaped-tensor build, test, baseline-retention and quality gates are required.")
    for model in ["minilm", "e5", "granite"]:
        for gate in ["parity", "batchAndConcurrency", "retrievalAndContracts", "tokenizerContract"]:
            if validation.get(model, {}).get(gate, {}).get("passed") is not True:
                raise RuntimeError(f"Cannot record final evidence: {model}/{gate} has not passed.")
    measurements = json.loads((ROOT / ".assets" / "measurements.json").read_text())
    if measurements.get("implementation") != "shaped-tensor-v1":
        raise RuntimeError("Current shaped-tensor measurements are required.")
    interop = {
        item["name"]: item
        for item in map(json.loads, (ROOT / ".assets" / "tensor-interop.jsonl").read_text().splitlines())
    }
    for name in ["stable-offset", "lexical-span-pin-offset", "stable-alias-gc",
                 "explicit-strided-copy", "stable-empty", "shape-zero", "singleton-span-control",
                 "stable-lifetime-normal", "stable-lifetime-exception",
                 "stable-lifetime-cancellation", "stable-lifetime-construction-error"]:
        if interop.get(name, {}).get("status") != "observed":
            raise RuntimeError(f"Stable interop control failed: {name}")
    for name in ["dense", "offset", "strided", "singleton", "empty", "shaped-empty"]:
        if interop.get(name + "-bridge", {}).get("error") != "MissingMethodException":
            raise RuntimeError(f"Pinned bridge observation changed; investigate before updating evidence: {name}")
    for name in ["python-validation.json", "dotnet-validation.json", "measurements.json",
                 "roberta-repro.jsonl", "bert-repro.jsonl", "tokenizer-contract-repro.jsonl",
                 "tensor-interop.jsonl"]:
        source = ROOT / ".assets" / name
        if not source.exists():
            raise FileNotFoundError(f"Required evidence has not been generated: {source}")
        shutil.copyfile(source, destination / name)
    provenance = [
        json.loads((ROOT / ".assets" / model / "provenance.json").read_text())
        for model in ["minilm", "e5", "granite"]
    ]
    (destination / "asset-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    projects = []
    for directory in ["src", "samples", "tools", "tests"]:
        for asset_file in sorted((ROOT / directory).glob("**/obj/project.assets.json")):
            assets = json.loads(asset_file.read_text(encoding="utf-8-sig"))
            direct = assets["project"]["frameworks"]["net10.0"].get("dependencies", {})
            packages = []
            for key, value in assets["targets"]["net10.0"].items():
                if value["type"] != "package":
                    continue
                name, version = key.rsplit("/", 1)
                if (name == "Microsoft.ML" or
                    (name.startswith("Microsoft.ML.") and not name.startswith(("Microsoft.ML.Tokenizers", "Microsoft.ML.OnnxRuntime"))) or
                    name.startswith(("Microsoft.SemanticKernel", "Microsoft.Agents.AI", "Microsoft.Extensions.VectorData"))):
                    raise RuntimeError(f"Forbidden dependency in {asset_file}: {name}")
                packages.append({"name": name, "version": version, "direct": name in direct})
            projects.append({"project": str(asset_file.parent.parent.relative_to(ROOT)), "packages": packages})
    if not projects:
        raise RuntimeError("No restored .NET projects found.")
    (destination / "dependencies.json").write_text(json.dumps(projects, indent=2) + "\n")


if __name__ == "__main__":
    main()
