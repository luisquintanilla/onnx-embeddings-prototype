"""Full, repeatable Linux CPU setup. Run as the devcontainer's non-root user."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv-linux"
ASSETS = ROOT / ".assets" / "devcontainer"
LOCK = ROOT / "dev" / "requirements-linux.lock.txt"


def run(*command, capture=False):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def check_environment():
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != VENV.resolve():
        raise RuntimeError(f"Expected the setup-owned environment at {VENV}.")
    for line in LOCK.read_text().splitlines():
        if line and not line.startswith("#") and "==" in line:
            name, pin = line.split("==")
            actual = importlib.metadata.version(name)
            if actual != pin:
                raise RuntimeError(f"{name}: expected {pin}, got {actual}. Move {VENV} aside explicitly and rerun setup.")
    import torch
    if torch.__version__ != "2.9.1+cpu" or torch.version.cuda is not None:
        raise RuntimeError("Expected CPU-only torch 2.9.1+cpu. Move .venv-linux aside explicitly and rerun setup.")
    if any(d.metadata["Name"].lower().startswith("nvidia-") for d in importlib.metadata.distributions()):
        raise RuntimeError("Unexpected CUDA packages in the CPU reference environment.")


def main():
    if sys.platform != "linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use the Linux x64 devcontainer with Python 3.12; Windows uses the existing .venv instructions.")
    if os.geteuid() == 0:
        raise RuntimeError("Run setup as the non-root vscode user, not root.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-environment", action="store_true", help="Check the existing Linux venv without installing anything.")
    args = parser.parse_args()
    if args.check_environment:
        check_environment()
        return
    sdk = json.loads((ROOT / "global.json").read_text())["sdk"]["version"]
    if run("dotnet", "--version", capture=True).strip() != sdk:
        raise RuntimeError(f"Expected SDK {sdk}; rebuild the devcontainer. Do not relax global.json.")

    lock_hash = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    marker = VENV / ".onnx-bootstrap.json"
    if VENV.exists():
        if VENV.is_symlink() or not marker.is_file() or not (VENV / "bin" / "python").is_file():
            raise RuntimeError(f"{VENV} is not a setup-owned Linux environment. Move it aside explicitly, then rerun; nothing was deleted.")
        state = json.loads(marker.read_text())
        if state["lockSha256"] != lock_hash:
            raise RuntimeError(f"The Linux dependency lock changed. Move {VENV} aside explicitly and rerun to create a new environment.")
        if not isinstance(state["installed"], bool):
            raise RuntimeError(f"Invalid setup marker in {VENV}; inspect the environment before retrying.")
    else:
        run(sys.executable, "-m", "venv", VENV)
        state = {"lockSha256": lock_hash, "installed": False}
        marker.write_text(json.dumps(state) + "\n")
    python = VENV / "bin" / "python"
    identity = json.loads(run(python, "-c",
        "import json,sys; print(json.dumps([sys.platform, list(sys.version_info[:2]), sys.prefix, sys.base_prefix]))",
        capture=True))
    if (identity[0] != "linux" or identity[1] != [3, 12]
            or Path(identity[2]).resolve() != VENV.resolve() or identity[2] == identity[3]):
        raise RuntimeError(f"{VENV} has the wrong interpreter. Move it aside explicitly; no packages were installed.")
    if not state["installed"]:
        # All direct/transitive packages are pinned; pip must not resolve extra dependencies.
        run(python, "-m", "pip", "install", "--no-input", "--disable-pip-version-check", "--only-binary=:all:",
            "--no-deps", "--index-url", "https://pypi.org/simple", "-r", LOCK)
    run(python, "-m", "pip", "check")
    run(python, __file__, "--check-environment")
    state["installed"] = True
    marker.write_text(json.dumps(state) + "\n")

    ASSETS.mkdir(parents=True, exist_ok=True)
    os.environ["ONNX_TEST_ASSET_ROOT"] = str(ASSETS)
    os.environ["ONNX_TEST_TOKENIZER_FIXTURES"] = str(ASSETS / "tokenizer-fixtures")
    run("dotnet", "restore", "OnnxEmbeddings.slnx", "--locked-mode")
    run("dotnet", "build", "OnnxEmbeddings.slnx", "--no-restore")
    run(python, "dev/acquire.py", "--asset-root", ASSETS)
    run(python, "dev/reference.py", "--asset-root", ASSETS)
    run(python, "dev/tokenizer_reference.py", "--asset-root", ASSETS,
        "--fixture-directory", ASSETS / "tokenizer-fixtures")
    run("dotnet", "test", "OnnxEmbeddings.slnx", "--no-build", "--no-restore")
    for sample in ("Providers", "Composition"):
        for model in ("minilm", "e5", "granite"):
            run("dotnet", "run", "--project", f"samples/{sample}", "--no-build", "--no-restore",
                "--", model, ASSETS / model)
    (ASSETS / "setup-environment.json").write_text(json.dumps({
        "sdk": sdk, "python": platform.python_version(), "platform": platform.platform(),
        "architecture": platform.machine(), "linuxLockSha256": lock_hash,
        "tests": "full suite passed", "samples": "all six passed",
    }, indent=2) + "\n")
    print("Ready: local CPU models, independent references, full tests and all six samples passed.", flush=True)


if __name__ == "__main__":
    main()
