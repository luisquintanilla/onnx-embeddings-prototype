"""Offline tokenizer-only oracles; never reads or writes model reference.json.

The committed fixtures are tiny untrained vocabularies, not publisher assets.
IDs/decode come from pinned HF/Python libraries, not the .NET implementation.
Run: .venv/Scripts/python.exe dev/tokenizer_reference.py
"""
import importlib.metadata
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "CommunityToolkit.Embeddings.Onnx.Tests" / "Fixtures"
PACKAGES = {"transformers": "4.57.6", "tokenizers": "0.22.2"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def cases(tokenizer, texts):
    result = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        result.append({"text": text, "ids": ids,
                       "tokens": tokenizer.convert_ids_to_tokens(ids),
                       "decoded": tokenizer.decode(ids, skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)})
    return result


def main():
    for package, pin in PACKAGES.items():
        assert importlib.metadata.version(package) == pin, f"Expected {package}=={pin}"
    bert_vocab = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", ",", "!",
        "cafe", "Hello", "Café", "play", "##ing", "中", "before", "after",
        "[", "]", "cls", "a", "##b", "##c", "hello!", "##!", "é", "😀",
        "supercalifragilisticexpialidocious",
    ]
    (FIXTURES / "bert-contract-vocab.txt").write_text("\n".join(bert_vocab) + "\n", encoding="utf-8")
    byte_vocab = {char: byte + 4 for byte, char in bytes_to_unicode().items()}
    byte_vocab.update({"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 50264})
    # Deliberately allow hello! as one merge only when ordinary pretokenization is off.
    merges = ["h e", "he l", "hel l", "hell o", "hello !", "X Ã", "© Y"]
    for index, merge in enumerate(merges):
        byte_vocab[merge.replace(" ", "")] = 260 + index
    write_json(FIXTURES / "granite-contract-vocab.json", byte_vocab)
    (FIXTURES / "granite-contract-merges.txt").write_text(
        "#version: 0.2\n" + "\n".join(merges) + "\n", encoding="utf-8")
    bert = BertTokenizer(vocab_file=str(FIXTURES / "bert-contract-vocab.txt"), do_lower_case=True)
    granite = RobertaTokenizer(vocab_file=str(FIXTURES / "granite-contract-vocab.json"),
                               merges_file=str(FIXTURES / "granite-contract-merges.txt"))
    synthetic = {
        "packages": PACKAGES,
        "bert": cases(bert, ["", "Hello, WORLD!", "Cafe\u0301\tHello", "playing",
                              "[CLS]Hello[SEP][PAD][UNK][MASK]", "Café 中", "a\x00b", "😀"]),
        "granite": cases(granite, ["", "hello!", "Aé😀Z", " \t\r\n",
                                    "<s>hello</s><pad><unk>  <mask>",
                                    "before \t\n<mask> after", "éé", "𐐀𝟘"]),
    }
    write_json(FIXTURES / "tokenizer-contract-goldens.json", synthetic)

    real = {"packages": PACKAGES, "models": {}}
    for key in ["minilm", "e5", "granite"]:
        folder = ROOT / ".assets" / key
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        real["models"][key] = cases(tokenizer, [
            "Hello, WORLD!", "Cafe\u0301\tHello", "é😀𐐀𝟘", " \t\r\n",
            "[CLS]Hello[SEP][PAD][UNK][MASK]", "<s>hello</s><pad><unk>  <mask>",
            "before \t\n<mask> after",
        ])
    write_json(ROOT / ".assets" / "tokenizer-contract-reference.json", real)
    print("Generated 16 synthetic and 21 local publisher tokenizer ID/decode cases; original reference.json untouched.")


if __name__ == "__main__":
    main()
