"""Pinned JevBench public-task diagnostics for submitted typed predictions.

This is a local public-subset scorecard, not JevBench's official ranking. It
never calls a model or sees the benchmark's sealed tasks.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

REVISION = "1bcc55eb6c8cffde2306b3db03ede39b61c6152a"
SOURCE = "https://github.com/fstandhartinger/jevbench"
FILES = {
    "original": (72, "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"),
    "easy": (48, "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"),
    "hard": (111, "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"),
}
STRICT_TOL = 0.001
ROUNDING_TOL = 0.02
MAX_SUBMISSION_BYTES = 8 * 1024 * 1024


def _file(tier: str, cache: Path) -> Path:
    _, expected = FILES[tier]
    destination = cache / f"{tier}.jsonl"
    cache.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not destination.is_symlink() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        return destination
    url = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{REVISION}/datasets/public/{tier}.jsonl"
    temporary = destination.with_suffix(".download")
    digest = hashlib.sha256()
    try:
        with urlopen(url, timeout=90) as response, temporary.open("wb") as output:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 2 * 1024 * 1024:
                    raise ValueError("JevBench public file exceeds pinned size limit")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("JevBench public file hash mismatch")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_public_tasks(cache: Path) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    for tier, (expected_count, _) in FILES.items():
        with _file(tier, cache).open("r", encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        if len(rows) != expected_count:
            raise ValueError("JevBench public task count changed")
        for row in rows:
            task_id, labels, question = row.get("id"), row.get("labels"), row.get("question")
            if (not isinstance(task_id, str) or task_id in tasks or row.get("split") != "public"
                    or not isinstance(labels, list) or not labels or any(not isinstance(x, str) for x in labels)
                    or len(set(labels)) != len(labels) or not isinstance(question, dict)
                    or question.get("type") not in {"noul", "choice", "score"}
                    or str(row.get("expected")) not in labels):
                raise ValueError("JevBench public task schema changed")
            row["tier"] = tier
            tasks[task_id] = row
    if len(tasks) != 231:
        raise ValueError("JevBench public task set changed")
    return tasks


def parse_submission(raw: bytes) -> list[dict]:
    if not raw or len(raw) > MAX_SUBMISSION_BYTES:
        raise ValueError("JevBench prediction JSONL must be nonempty and at most 8 MiB")
    rows = []
    seen = set()
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise ValueError(f"prediction line {number} needs a task_id")
        if row["task_id"] in seen:
            raise ValueError(f"duplicate task_id on prediction line {number}")
        seen.add(row["task_id"])
        rows.append(row)
        if len(rows) > 231:
            raise ValueError("submission exceeds 231 public tasks")
    if not rows:
        raise ValueError("JevBench prediction JSONL is empty")
    return rows


def _score(row: dict, task: dict) -> dict:
    labels = task["labels"]
    gold = str(task["expected"])
    if row.get("ok") is False:
        return {"valid": False, "strict_valid": False, "correct": False, "probs": None}
    # Official runner rows preserve the as-returned vector separately. Recheck
    # it instead of trusting its computed 'valid' or 'correct' fields.
    probabilities = (row["probs_as_returned"] if row.get("probs_as_returned") is not None
                     else row.get("probabilities", row.get("probs")))
    if probabilities is None:
        label = row.get("label")
        if row.get("probs_source") == "label_only_no_calibrated_distribution":
            label = row.get("predicted")
        valid = isinstance(label, str) and label in labels
        return {"valid": valid, "strict_valid": valid, "correct": valid and label == gold, "probs": None}
    if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
        return {"valid": False, "strict_valid": False, "correct": False, "probs": None}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           or not 0 <= value <= 1 for value in probabilities.values()):
        return {"valid": False, "strict_valid": False, "correct": False, "probs": None}
    total = sum(probabilities.values())
    if abs(total - 1) > ROUNDING_TOL or total <= 0:
        return {"valid": False, "strict_valid": False, "correct": False, "probs": None}
    strict = abs(total - 1) <= STRICT_TOL
    probs = {key: float(value) / total if not strict else float(value) for key, value in probabilities.items()}
    predicted = min(labels, key=lambda label: (-probs[label], label))
    return {"valid": True, "strict_valid": strict, "correct": predicted == gold,
            "probs": probs, "confidence": max(probs.values())}


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _ece(pairs: list[tuple[float, bool]]) -> float | None:
    if not pairs:
        return None
    bins = [[] for _ in range(10)]
    for confidence, correct in pairs:
        bins[min(9, int(confidence * 10))].append((confidence, correct))
    return sum(len(bucket) / len(pairs) * abs(statistics.mean(conf for conf, _ in bucket)
                                             - statistics.mean(float(correct) for _, correct in bucket))
               for bucket in bins if bucket)


def score_public_submission(tasks: dict[str, dict], rows: list[dict]) -> dict:
    submitted = {}
    for row in rows:
        task_id = row["task_id"]
        if task_id not in tasks:
            raise ValueError(f"unknown JevBench public task_id: {task_id[:80]}")
        if task_id in submitted:
            raise ValueError("duplicate JevBench public task_id")
        submitted[task_id] = row
    totals = Counter()
    by_tier = {tier: Counter() for tier in FILES}
    by_type = {kind: Counter() for kind in ("noul", "choice", "score")}
    pairs, briers, latency = [], [], []
    for task_id, task in tasks.items():
        if task.get("provenance", {}).get("exclude_reason") or task.get("expected") is None:
            continue
        tier, kind = task["tier"], task["question"]["type"]
        for counts in (totals, by_tier[tier], by_type[kind]):
            counts["planned"] += 1
        row = submitted.get(task_id)
        if row is None:
            continue
        scored = _score(row, task)
        for counts in (totals, by_tier[tier], by_type[kind]):
            counts["submitted"] += 1
            counts["valid"] += scored["valid"]
            counts["strict_valid"] += scored["strict_valid"]
            counts["correct"] += scored["correct"]
        if scored["probs"] is not None:
            gold = str(task["expected"])
            pairs.append((scored["confidence"], scored["correct"]))
            briers.append(sum((scored["probs"][label] - int(label == gold)) ** 2 for label in task["labels"]))
        timing = row.get("latency_ms")
        if timing is None and "latency_s" in row:
            timing = row["latency_s"] * 1000 if isinstance(row["latency_s"], (int, float)) else None
        if isinstance(timing, (int, float)) and not isinstance(timing, bool) and math.isfinite(timing) and timing >= 0:
            latency.append(float(timing))
    def summary(counts: Counter) -> dict:
        return {"planned": counts["planned"], "submitted": counts["submitted"],
                "valid": counts["valid"], "strict_valid": counts["strict_valid"],
                "correct": counts["correct"],
                "accuracy_full_denominator": counts["correct"] / counts["planned"] if counts["planned"] else None}
    return {"schema_version": "1.0", "scope": "jevbench_public_subset_diagnostic",
            "official_jevbench_score": None, "source": SOURCE, "source_revision": REVISION,
            "source_files_sha256": {tier: digest for tier, (_, digest) in FILES.items()},
            "overall": summary(totals), "by_tier": {key: summary(value) for key, value in by_tier.items()},
            "by_type": {key: summary(value) for key, value in by_type.items()},
            "calibration_rows": len(pairs), "brier_mean_sum": statistics.mean(briers) if briers else None,
            "ece_10_bins": _ece(pairs), "timing_rows": len(latency),
            "reported_p50_ms": _percentile(latency, 0.5), "reported_p95_ms": _percentile(latency, 0.95),
            "limitations": ["Public tasks only; no sealed tasks, official composite score, measured cost, or independent model-timing verification.",
                            "Prediction-file timing is self-reported. Public items may have influenced model development."]}
