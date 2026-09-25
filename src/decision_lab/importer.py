"""Bounded CSV/JSONL import with row-level issues and conflict quarantine.

This module deliberately does not train a model. Its clean output is a candidate
dataset that still needs domain-owner review and a frozen split plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_ROWS = 50_000
MAX_ROW_BYTES = 1 * 1024 * 1024
LABEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FIELDS = frozenset(
    {"id", "text", "label", "group_id", "occurred_at", "source", "label_origin", "label_policy_version"}
)
OPTIONAL_FIELDS = FIELDS - {"id", "text", "label"}


class ImportErrorDetail(ValueError):
    """An input file or label schema cannot be imported."""


@dataclass(frozen=True)
class ImportResult:
    records: list[dict[str, str]]
    report: dict[str, Any]


def _validate_labels(labels: list[str]) -> None:
    if not isinstance(labels, list) or not 2 <= len(labels) <= 20:
        raise ImportErrorDetail("labels must be a list of 2 to 20 IDs")
    if any(not isinstance(label, str) or not LABEL_ID.fullmatch(label) for label in labels):
        raise ImportErrorDetail("label IDs must be nonempty ASCII letters, digits, _, . or -")
    if len(set(labels)) != len(labels):
        raise ImportErrorDetail("label IDs must be unique")


def _issue(issues: list[dict[str, Any]], row: int, reason: str, row_id: str | None = None) -> None:
    issue: dict[str, Any] = {"row": row, "reason": reason}
    if row_id is not None:
        issue["id"] = row_id
    issues.append(issue)


def _read_jsonl(raw: bytes, issues: list[dict[str, Any]]) -> list[tuple[int, Any]]:
    rows: list[tuple[int, Any]] = []
    for number, line in enumerate(raw.splitlines(), 1):
        if number > MAX_ROWS:
            raise ImportErrorDetail(f"row limit exceeded: {MAX_ROWS}")
        if len(line) > MAX_ROW_BYTES:
            _issue(issues, number, "row_too_large")
            continue
        try:
            decoded = line.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            _issue(issues, number, "invalid_utf8")
            continue
        if not decoded.strip():
            _issue(issues, number, "empty_row")
            continue
        try:
            rows.append((number, json.loads(decoded)))
        except json.JSONDecodeError:
            _issue(issues, number, "malformed_json")
    return rows


def _read_csv(raw: bytes, issues: list[dict[str, Any]]) -> list[tuple[int, Any]]:
    try:
        decoded = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ImportErrorDetail("CSV contains invalid UTF-8") from exc
    if any(len(line.encode("utf-8")) > MAX_ROW_BYTES for line in decoded.splitlines()):
        raise ImportErrorDetail("CSV contains a physical line larger than the row limit")
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(MAX_ROW_BYTES)
        reader = csv.DictReader(io.StringIO(decoded, newline=""), restkey="__extra__", strict=True)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)) or not {"text", "label"} <= set(headers):
            raise ImportErrorDetail("CSV needs unique headers including text and label")
        if set(headers) - FIELDS:
            raise ImportErrorDetail("CSV has unsupported columns; review decision-time fields before import")
        rows = []
        for number, row in enumerate(reader, 1):
            if number > MAX_ROWS:
                raise ImportErrorDetail(f"row limit exceeded: {MAX_ROWS}")
            if "__extra__" in row or any(value is None for value in row.values()):
                _issue(issues, number, "malformed_csv_row")
            elif sum(len(value.encode("utf-8")) for value in row.values()) > MAX_ROW_BYTES:
                _issue(issues, number, "row_too_large")
            else:
                rows.append((number, row))
        return rows
    except csv.Error as exc:
        raise ImportErrorDetail(f"malformed CSV: {exc}") from exc
    finally:
        csv.field_size_limit(previous_limit)


def validate_file(path: Path, labels: list[str]) -> ImportResult:
    """Validate a bounded file and return clean records plus a text-free issue report."""
    _validate_labels(labels)
    if path.suffix.lower() not in {".csv", ".jsonl"}:
        raise ImportErrorDetail("input must end in .csv or .jsonl")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ImportErrorDetail(f"upload limit exceeded: {MAX_UPLOAD_BYTES} bytes")
    raw = path.read_bytes()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ImportErrorDetail(f"upload limit exceeded: {MAX_UPLOAD_BYTES} bytes")
    issues: list[dict[str, Any]] = []
    rows = _read_csv(raw, issues) if path.suffix.lower() == ".csv" else _read_jsonl(raw, issues)
    label_set = set(labels)
    seen_ids: set[str] = set()
    candidates: list[tuple[int, dict[str, str]]] = []

    for number, original in rows:
        if not isinstance(original, dict):
            _issue(issues, number, "record_not_object")
            continue
        if set(original) - FIELDS:
            _issue(issues, number, "unsupported_fields")
            continue
        row_id = original.get("id")
        if row_id is None:
            row_id = f"row-{number:06d}"
        if not isinstance(row_id, str) or not row_id.strip():
            _issue(issues, number, "invalid_id")
            continue
        if row_id in seen_ids:
            _issue(issues, number, "duplicate_id", row_id)
            continue
        item_text = original.get("text")
        if not isinstance(item_text, str) or not item_text.strip():
            _issue(issues, number, "empty_or_invalid_text", row_id)
            continue
        label = original.get("label")
        if not isinstance(label, str) or label not in label_set:
            _issue(issues, number, "unknown_label", row_id)
            continue
        if any(value is not None and (not isinstance(value, str) or not value.strip()) for key, value in original.items() if key in OPTIONAL_FIELDS):
            _issue(issues, number, "invalid_metadata", row_id)
            continue
        clean = {"id": row_id, "text": item_text, "label": label}
        clean.update({key: original[key] for key in OPTIONAL_FIELDS if original.get(key) is not None})
        seen_ids.add(row_id)
        candidates.append((number, clean))

    by_text: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row in candidates:
        by_text[row[1]["text"]].append(row)
    conflicted = {
        item_text for item_text, group in by_text.items() if len({record["label"] for _, record in group}) > 1
    }
    for item_text in conflicted:
        for number, record in by_text[item_text]:
            _issue(issues, number, "conflicting_labels_for_identical_text", record["id"])
    clean_records = [record for _, record in candidates if record["text"] not in conflicted]
    counts = Counter(record["label"] for record in clean_records)
    missing = {key: sum(key not in record for record in clean_records) for key in sorted(OPTIONAL_FIELDS)}
    duplicate_groups = sum(len(group) > 1 for text, group in by_text.items() if text not in conflicted)
    report = {
        "schema_version": "1.0",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "input_rows": len(rows) + sum(issue["reason"] in {"row_too_large", "invalid_utf8", "empty_row", "malformed_json", "malformed_csv_row"} for issue in issues),
        "accepted_rows": len(clean_records),
        "rejected_rows": len({issue["row"] for issue in issues}),
        "class_counts": {label: counts[label] for label in labels},
        "missing_metadata_counts": missing,
        "exact_duplicate_groups": duplicate_groups,
        "conflicting_text_groups": len(conflicted),
        "issues": sorted(issues, key=lambda issue: issue["row"]),
        "requires_decision_time_leakage_review": True,
        "requires_input_length_review": True,
    }
    return ImportResult(clean_records, report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an MVP routing dataset")
    parser.add_argument("input", type=Path, help="CSV or JSONL file")
    parser.add_argument("--labels", type=Path, required=True, help="JSON array of stable label IDs")
    parser.add_argument("--report", type=Path, required=True, help="JSON issue report path")
    parser.add_argument("--clean", type=Path, required=True, help="accepted JSONL output path")
    args = parser.parse_args()
    if len({args.input.resolve(), args.labels.resolve(), args.report.resolve(), args.clean.resolve()}) != 4:
        parser.exit(2, "input, labels, report, and clean must be distinct files\n")
    if args.report.exists() or args.clean.exists():
        parser.exit(2, "report and clean output files must not already exist\n")
    try:
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        result = validate_file(args.input, labels)
    except (OSError, UnicodeError, json.JSONDecodeError, ImportErrorDetail) as exc:
        parser.exit(2, f"import failed: {exc}\n")
    try:
        with args.report.open("x", encoding="utf-8") as output:
            output.write(json.dumps(result.report, indent=2, ensure_ascii=False) + "\n")
        with args.clean.open("x", encoding="utf-8") as output:
            for record in result.records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        parser.exit(2, f"cannot write import output: {exc}\n")
    print(f"accepted {result.report['accepted_rows']} rows; rejected {result.report['rejected_rows']} rows")


if __name__ == "__main__":
    main()
