"""Generate offline HF/PyTorch Float32 goldens and inspect the exact publisher ONNX graphs."""
import json
import platform
from pathlib import Path
import importlib.metadata
import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def main():
    results = []
    for recipe in json.loads((ROOT / "dev" / "models.json").read_text()):
        folder = ROOT / ".assets" / recipe["key"]
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(folder, local_files_only=True, trust_remote_code=False,
                                         dtype=torch.float32, attn_implementation="eager").eval()
        graph = onnx.load(folder / "model.onnx")
        onnx.checker.check_model(graph)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(folder / "model.onnx"), options, providers=["CPUExecutionProvider"])
        output_name = "logits" if recipe["key"] == "granite" else "last_hidden_state"
        inputs = [
            "", " ", " \t\r\n", "hello", "Hello, WORLD!", "  leading spaces", "trailing spaces  ",
            "Caf\u00e9 na\u00efve r\u00e9sum\u00e9", "Cafe\u0301", "\u4e2d\u6587 \u65e5\u672c\u8a9e",
            "\U0001f600\U0001f680 \U0001f469\u200d\U0001f4bb", "\U00010400\U00010428 \U0001d7d8\U0001d7d9",
            "can't I'M we've...?! -- 123.45", "a\u00a0b\u2003c\u202fd", "a\x00b\ufffdc",
            "<s>one</s><pad><unk>  <mask> next", "before \t\n<mask> after",
            "[CLS]Hello[SEP][PAD][UNK][MASK]", "query: caller supplied passage: text",
            "a\tb\nc\rd", "x\u200dy\u200ez", "a\ue000b", "[cls] [mask] [CLS] [MASK]",
            "\u0130 \u212b \u03a3\u03a3", "\ud55c\uad6d\uc5b4", "a\u3400b\u4e00c",
            "a\x1cb\x85c", "a\U0002b820b\U0002b920c",
            "A dog is playing in the park.", "A puppy plays outdoors.", "Database indexes accelerate queries."
        ]
        if recipe["key"] == "e5":
            cases = [{"text": text, "purpose": role} for text in inputs for role in ["Query", "Document"]]
        else:
            cases = [{"text": text, "purpose": None} for text in inputs]
        # "hello" is one content token for both approved tokenizer vocabularies, after the first word.
        for count in [recipe["max_length"] - 3, recipe["max_length"] - 2,
                      recipe["max_length"] - 1, recipe["max_length"] + 8]:
            cases.append({"text": " ".join(["hello"] * count), "purpose": "Query" if recipe["key"] == "e5" else None})
        cases.append({"text": " ".join(["hello"] * (recipe["max_length"] - 3)) + " \U0001f600",
                      "purpose": "Document" if recipe["key"] == "e5" else None})
        batches = []
        max_error = 0.0
        for start in range(0, len(cases), 6):
            items = cases[start:start + 6]
            texts = [("query: " if item["purpose"] == "Query" else "passage: ") + item["text"]
                     if item["purpose"] else item["text"] for item in items]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=recipe["max_length"], return_tensors="pt")
            with torch.inference_mode():
                hidden = model(**tokens).last_hidden_state
                if recipe["pooling"] == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = tokens["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()
            native = session.run([output_name], {v.name: tokens[v.name].numpy() for v in session.get_inputs()})[0]
            if recipe["pooling"] == "cls":
                pooled = native[:, 0]
            else:
                mask = tokens["attention_mask"].numpy().astype(np.float32)[..., None]
                pooled = (native * mask).sum(1) / mask.sum(1)
            native_vectors = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
            error = float(np.max(np.abs(native_vectors - vectors)))
            max_error = max(max_error, error)
            if error > 2e-5:
                raise RuntimeError(f"{recipe['key']} publisher ONNX differs from PyTorch: {error}")
            batches.append({"items": items, "inputIds": tokens["input_ids"].tolist(),
                            "attentionMask": tokens["attention_mask"].tolist(),
                            "tokenTypeIds": tokens.get("token_type_ids", torch.zeros_like(tokens["input_ids"])).tolist(),
                            "vectors": vectors.tolist()})
        reference = {"model": recipe, "batches": batches}
        (folder / "reference.json").write_text(json.dumps(reference, ensure_ascii=True, separators=(",", ":")) + "\n")
        # Content-only goldens separate surrounding-token and padding policy from the BPE reproduction.
        repro = ["hello", "\u00e9", "\U0001f600", "hello world", "  hello", "a\tb", "<mask>"]
        (folder / "tokenizer-probes.json").write_text(json.dumps(
            [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False)} for text in repro], indent=2) + "\n")
        evidence = {"key": recipe["key"], "revision": recipe["revision"], "cases": len(cases),
                    "maxPythonOnnxVsTorchAbsoluteError": max_error,
                    "inputs": [{"name": v.name, "type": v.type, "shape": v.shape} for v in session.get_inputs()],
                    "outputs": [{"name": v.name, "type": v.type, "shape": v.shape} for v in session.get_outputs()],
                    "selectedTokenOutput": output_name, "pooling": recipe["pooling"],
                    "onnxProducer": graph.producer_name, "onnxProducerVersion": graph.producer_version,
                    "opsets": [{"domain": o.domain, "version": o.version} for o in graph.opset_import]}
        results.append(evidence)
        print(json.dumps(evidence), flush=True)
    report = {"python": platform.python_version(), "packages": {
        name: importlib.metadata.version(name) for name in
        ["numpy", "onnx", "onnxruntime", "safetensors", "torch", "transformers", "tokenizers"]
    }, "models": results}
    (ROOT / ".assets" / "python-validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
