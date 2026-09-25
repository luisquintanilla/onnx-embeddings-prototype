"""Explicit developer acquisition only. Nothing in the .NET library calls this."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS = json.loads((ROOT / "dev" / "models.json").read_text())


def main():
    recorded = ROOT / "docs" / "evidence" / "asset-provenance.json"
    expected = {}
    if recorded.exists():
        for model in json.loads(recorded.read_text()):
            for file in model["files"]:
                expected[(model["repository"], model["revision"], file["source"])] = file["sha256"]
    for model in MODELS:
        folder = ROOT / ".assets" / model["key"]
        folder.mkdir(parents=True, exist_ok=True)
        files = [
            "config.json", "tokenizer_config.json", "tokenizer.json",
            "special_tokens_map.json", "sentence_bert_config.json",
            "modules.json", "1_Pooling/config.json", "model.safetensors",
            *model["tokenizer_files"], model["onnx"],
        ]
        evidence = {"repository": model["repository"], "revision": model["revision"], "files": []}
        for name in files:
            url = f'https://huggingface.co/{model["repository"]}/resolve/{model["revision"]}/{name}'
            target = folder / ("model.onnx" if name == model["onnx"] else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                print(f"Downloading {model['key']}/{name}", flush=True)
                temporary = target.with_suffix(target.suffix + ".partial")
                request = urllib.request.Request(url, headers={"User-Agent": "local-onnx-embeddings-validation"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                temporary.replace(target)
            with target.open("rb") as content:
                digest = hashlib.file_digest(content, "sha256").hexdigest()
            known = expected.get((model["repository"], model["revision"], name))
            if known is not None and digest != known:
                raise RuntimeError(f"SHA-256 mismatch for {target}; the local file is not the verified pinned artifact.")
            evidence["files"].append({"source": name, "url": url, "bytes": target.stat().st_size, "sha256": digest})
        (folder / "provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"Recorded {model['key']} hashes", flush=True)


if __name__ == "__main__":
    main()
