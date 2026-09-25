"""Local shadow-mode batch inference; never executes a routing action."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .model import DecisionModel, InvalidInputError

MAX_ROWS = 50_000
MAX_FILE_BYTES = 100 * 1024 * 1024


def shadow_run(bundle: Path, requests: Path, output: Path, *, trusted_public_key: bytes | None = None,
               evaluation: bool = False) -> dict:
    if output.exists() or output.is_symlink() or output.resolve() == requests.resolve():
        raise ValueError("shadow output must be a new file")
    if requests.is_symlink() or not requests.is_file() or requests.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("shadow input must be a regular JSONL file within the size limit")
    model = DecisionModel.load(bundle, allow_unsigned=trusted_public_key is None,
                               trusted_public_key=trusted_public_key)
    counts = Counter()
    seen_ids = set()
    try:
        with requests.open("r", encoding="utf-8") as source, output.open("x", encoding="utf-8") as destination:
            for line_number, line in enumerate(source, start=1):
                if line_number > MAX_ROWS:
                    raise ValueError("shadow input exceeds row limit")
                if len(line.encode("utf-8")) > 128 * 1024:
                    raise ValueError(f"shadow row {line_number} exceeds size limit")
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on shadow row {line_number}") from exc
                if not isinstance(request, dict) or not isinstance(request.get("id"), str) or not request["id"] or not isinstance(request.get("text"), str):
                    raise ValueError(f"shadow row {line_number} needs id and text")
                if request["id"] in seen_ids:
                    raise ValueError(f"duplicate shadow id on row {line_number}")
                seen_ids.add(request["id"])
                try:
                    prediction = model.predict(request["text"], evaluation=evaluation)
                    result = {"id": request["id"], **prediction.to_dict()}
                    counts[prediction.status] += 1
                    if prediction.abstention_reason:
                        counts["reason:" + prediction.abstention_reason] += 1
                except InvalidInputError:
                    result = {"id": request["id"], "error_code": "invalid_input"}
                    counts["invalid_input"] += 1
                destination.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {"rows": len(seen_ids), "counts": dict(counts), "output": str(output),
            "mode": "evaluation" if evaluation else "shadow"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local decision bundle in shadow mode")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("requests", type=Path, help="JSONL records with id and text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-public-key", type=Path)
    parser.add_argument("--evaluation", action="store_true", help="report would-accept outcomes without actionable choices")
    args = parser.parse_args()
    try:
        result = shadow_run(args.bundle, args.requests, args.output,
                            trusted_public_key=args.trusted_public_key.read_bytes() if args.trusted_public_key else None,
                            evaluation=args.evaluation)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"shadow run failed: {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
