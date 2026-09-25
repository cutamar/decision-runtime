"""Validate a bundle and print safe, text-free metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import DecisionModel, sha256


def inspect_bundle(bundle: Path, *, trusted_public_key: bytes | None = None) -> dict:
    model = DecisionModel.load(bundle, allow_unsigned=trusted_public_key is None,
                               trusted_public_key=trusted_public_key)
    return {
        "format_version": model.manifest["format_version"],
        "artifact_kind": model.manifest["artifact_kind"],
        "model_version": model.manifest["model_version"],
        "policy_version": model.manifest["policy_version"],
        "trust": model.manifest["trust"],
        "manifest_sha256": sha256((bundle / "manifest.json").read_bytes()),
        "inference_payload_sha256": model.manifest["inference_payload_sha256"],
        "labels": model.labels,
        "calibration_status": model.calibration["status"],
        "qualification_status": model.qualification["status"],
        "max_tokens": model.preprocessing["max_tokens"],
        "bundle_bytes": sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and inspect a local decision bundle")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--trusted-public-key", type=Path)
    args = parser.parse_args()
    try:
        key = args.trusted_public_key.read_bytes() if args.trusted_public_key else None
        result = inspect_bundle(args.bundle, trusted_public_key=key)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"inspection failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
