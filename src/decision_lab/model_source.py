"""Immutable source manifest for the approved MiniLM feasibility candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "bc57282bc374d33e0d6c4de27f12dc1c2a87f37a"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"
SOURCE_FILES = {
    "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    "README.md": "766a000da416a8a45760fce75dd387cc2ba3a358f7c58b4813a7b222ddb32471",
}
ADAPTATION_WEIGHTS_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"


def verify_source(root: Path) -> None:
    for name, expected in SOURCE_FILES.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or linked source file: {name}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"source hash mismatch: {name}")


def candidate_source_info(root: Path) -> dict:
    """Verify a frozen source or a locally adapted export and return its lineage."""
    adaptation_path = root / "adaptation.json"
    if not adaptation_path.exists():
        verify_source(root)
        return {"recipe": "frozen_encoder", "base_model_id": MODEL_ID,
                "base_model_revision": REVISION, "source_file_hashes": SOURCE_FILES}
    if adaptation_path.is_symlink():
        raise ValueError("linked adaptation manifest is unsupported")
    info = json.loads(adaptation_path.read_text(encoding="utf-8"))
    if (info.get("recipe") != "last_layer_supervised_v1" or info.get("base_model_id") != MODEL_ID
            or info.get("base_model_revision") != REVISION
            or info.get("base_model_weights_sha256") != ADAPTATION_WEIGHTS_SHA256):
        raise ValueError("unsupported adapted source")
    parity = info.get("max_conversion_abs_diff")
    samples = info.get("conversion_parity_samples")
    if (not isinstance(parity, (int, float)) or not 0 <= parity <= 1e-3
            or not isinstance(samples, list) or not any(item.get("tokens") == 256 for item in samples if isinstance(item, dict))):
        raise ValueError("adapted source lacks passing conversion parity evidence")
    hashes = info.get("source_file_hashes")
    if not isinstance(hashes, dict) or set(hashes) != {"onnx/model.onnx", "tokenizer.json"}:
        raise ValueError("incomplete adapted source hashes")
    for name, expected in hashes.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or linked adapted source: {name}")
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"adapted source hash mismatch: {name}")
    return info


def download_source(root: Path, *, include_weights: bool = False) -> None:
    files = dict(SOURCE_FILES)
    if include_weights:
        files["model.safetensors"] = ADAPTATION_WEIGHTS_SHA256
    for name, expected in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        temporary = path.with_name(path.name + ".part")
        digest = hashlib.sha256()
        try:
            with urlopen(BASE_URL + name, timeout=180) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"download hash mismatch: {name}")
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
    verify_source(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned MiniLM ONNX candidate")
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-weights", action="store_true", help="also download verified safetensors for adaptation")
    args = parser.parse_args()
    try:
        download_source(args.output, include_weights=args.include_weights)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"model download failed: {exc}\n")
    print(f"verified {MODEL_ID}@{REVISION} in {args.output}")


if __name__ == "__main__":
    main()
