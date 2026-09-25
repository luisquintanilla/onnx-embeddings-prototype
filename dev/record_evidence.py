"""Copy generated validation evidence and inventory resolved .NET dependencies; never copy model assets."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "docs" / "evidence"
    destination.mkdir(parents=True, exist_ok=True)
    validation = json.loads((ROOT / ".assets" / "dotnet-validation.json").read_text())
    for model in ["minilm", "e5", "granite"]:
        for gate in ["parity", "batchAndConcurrency", "retrievalAndContracts"]:
            if validation.get(model, {}).get(gate, {}).get("passed") is not True:
                raise RuntimeError(f"Cannot record final evidence: {model}/{gate} has not passed.")
    for name in ["python-validation.json", "dotnet-validation.json", "measurements.json", "roberta-repro.jsonl", "bert-repro.jsonl"]:
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
