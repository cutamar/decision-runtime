"""Local labeled evaluation through the exact exported inference kernel."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from .model import DecisionModel, InvalidInputError, sha256

MAX_ROWS = 50_000
MAX_FILE_BYTES = 100 * 1024 * 1024


def _rows(path: Path, labels: set[str]):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("labeled input must be a regular file within 100 MiB")
    if path.suffix.lower() not in {".csv", ".jsonl"}:
        raise ValueError("labeled input must be CSV or JSONL")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(source, strict=True)
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or not {"text", "label"} <= set(reader.fieldnames):
                raise ValueError("CSV needs unique headers including text and label")
            records = reader
        else:
            def jsonl():
                for line in source:
                    if line.strip():
                        yield json.loads(line)
            records = jsonl()
        seen = set()
        count = 0
        for row in records:
            count += 1
            if count > MAX_ROWS:
                raise ValueError("labeled input exceeds 50,000 rows")
            if (not isinstance(row, dict) or not isinstance(row.get("text"), str)
                    or not row["text"].strip() or not isinstance(row.get("label"), str)
                    or row["label"] not in labels):
                raise ValueError(f"invalid text or label on row {count}")
            row_id = row.get("id")
            if row_id is not None:
                if not isinstance(row_id, str) or not row_id or row_id in seen:
                    raise ValueError(f"invalid or duplicate id on row {count}")
                seen.add(row_id)
            yield row
        if count == 0:
            raise ValueError("labeled input is empty")


def evaluate(bundle: Path, labeled: Path, *, trusted_public_key: bytes | None = None) -> dict:
    """Report descriptive metrics; no independence or qualification claim is made."""
    model = DecisionModel.load(bundle, allow_unsigned=trusted_public_key is None,
                               trusted_public_key=trusted_public_key)
    labels = model.labels
    confusion = {truth: {guess: 0 for guess in labels} for truth in labels}
    counts = Counter()
    reasons = Counter()
    log_losses = []
    brier = []
    bins = [{"count": 0, "confidence_sum": 0.0, "correct": 0} for _ in range(10)]
    for row in _rows(labeled, set(labels)):
        try:
            prediction = model.predict(row["text"], evaluation=True)
        except InvalidInputError as exc:
            raise ValueError("invalid request text in labeled input") from exc
        counts["rows"] += 1
        if prediction.abstention_reason:
            reasons[prediction.abstention_reason] += 1
        if prediction.probabilities is None:
            continue
        counts["supported"] += 1
        truth = row["label"]
        guess = prediction.suggested_choice
        confusion[truth][guess] += 1
        counts["correct"] += int(guess == truth)
        probs = prediction.probabilities
        log_losses.append(-math.log(max(probs[truth], 1e-15)))
        brier.append(sum((probs[label] - int(label == truth)) ** 2 for label in labels))
        confidence = prediction.top_probability
        bucket = bins[min(int(confidence * 10), 9)]
        bucket["count"] += 1
        bucket["confidence_sum"] += confidence
        bucket["correct"] += int(guess == truth)
        if prediction.status == "would_accept":
            counts["accepted"] += 1
            counts["accepted_errors"] += int(guess != truth)
    per_label = {}
    f1_values = []
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[truth][label] for truth in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    calibration_bins = [
        {"lower": index / 10, "upper": (index + 1) / 10, "count": item["count"],
         "mean_confidence": item["confidence_sum"] / item["count"] if item["count"] else None,
         "accuracy": item["correct"] / item["count"] if item["count"] else None}
        for index, item in enumerate(bins)
    ]
    supported = counts["supported"]
    accepted = counts["accepted"]
    return {
        "schema_version": "1.0",
        "mode": "descriptive_labeled_evaluation",
        "qualification_status": model.qualification["status"],
        "manifest_sha256": sha256((bundle / "manifest.json").read_bytes()),
        "labeled_file_sha256": sha256(labeled.read_bytes()),
        "labels": labels,
        "rows": counts["rows"], "supported": supported, "accepted": accepted,
        "coverage": accepted / counts["rows"],
        "supported_accuracy": counts["correct"] / supported if supported else None,
        "macro_f1": sum(f1_values) / len(labels),
        "accepted_errors": counts["accepted_errors"],
        "accepted_error_rate": counts["accepted_errors"] / accepted if accepted else None,
        "negative_log_likelihood": sum(log_losses) / supported if supported else None,
        "multiclass_brier_mean_sum": sum(brier) / supported if supported else None,
        "ece_10_bins": sum(item["count"] * abs(item["accuracy"] - item["mean_confidence"])
                           for item in calibration_bins if item["count"]) / supported if supported else None,
        "calibration_bins": calibration_bins,
        "per_label": per_label,
        "confusion_matrix": [[confusion[truth][guess] for guess in labels] for truth in labels],
        "abstention_reasons": dict(reasons),
        "limitations": ["Descriptive metrics only; row independence, label quality, representativeness, and untouched holdout status are not verified."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local bundle on labeled CSV or JSONL")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("labeled", type=Path)
    parser.add_argument("--trusted-public-key", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.output and (args.output.exists() or args.output.resolve() == args.labeled.resolve()):
            raise ValueError("output must be a new file distinct from the input")
        key = args.trusted_public_key.read_bytes() if args.trusted_public_key else None
        result = evaluate(args.bundle, args.labeled, trusted_public_key=key)
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        parser.exit(2, f"evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
