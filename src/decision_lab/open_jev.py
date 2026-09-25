"""Pinned, task-specific projections of public Open-Jev controls.

These projections are new fixed-label experiments, not JevBench or original
Open-Jev model scores. No target or audit metadata enters model input.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from .splits import PARTITIONS, dataset_digest, validate_split

DATASET_REPO = "ZefanCai/Open-Jev"
DATASET_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
SPLITS = ("train", "calibration", "validation", "test", "ood")
PRESETS = {
    "mail_kind_en": {
        "name": "Open-Jev mailroom · English email kind",
        "config": "mailroom-control-v1",
        "head": "kind",
        "labels": ["account_statement", "invoice", "newsletter", "other", "payment_confirmation", "promotion"],
        "license": "CC0-1.0 original synthetic mailroom controls. Source: https://huggingface.co/datasets/ZefanCai/Open-Jev/blob/main/cards/mailroom-control-v1.md",
        "hashes": {
            "train": "ebbfae0b4024ac8dbbbc709f54b8cc02023d5953a0b712888a9d2ffbb5189811",
            "calibration": "8cd86ca00c6d75b3ee356e04d8ca872f6442e5a0566e5a50c9af18709a6a130f",
            "validation": "5705a3babf6b45a231946e48a4a2eed41c81e8d5ec9de1de42ec69801da34db4",
            "test": "d6662f7f7870d3bc629956f07fed8dfb5f0a42d1a923fbbe764db30e933036be",
            "ood": "e63f4949c9103b26a718e7df2a4416161f46093e27569a34de64083c587dd941",
        },
    },
    "support_routing_en": {
        "name": "Open-Jev customer control · English support routing",
        "config": "release-v2-redistributable",
        "head": "category",
        "labels": ["account", "billing", "bug_report", "feature_request"],
        "license": "Derived from Open-Jev's public synthetic customer-control-v1 state and hard targets. Original generated records are marked CC0-1.0; the source card notes unverified rights for short upstream question descriptions, which this projection does not copy. https://huggingface.co/datasets/ZefanCai/Open-Jev",
        "hashes": {
            "train": "e67c8aa8b31f341981d78c4da5fc07a195a6d1357ee37ca9d7fc5b2135d3fde5",
            "calibration": "e46e9f0364253594bd63681c21de380970e8b740829deac8bafbd515e9de03c8",
            "validation": "dedc81c6674dbb068ded125f9bea04b9427a8f6d285909879562bcd4603216b4",
            "test": "a78bb20d4a79619f7ca5fcaf9283910cf9acbc5856ec9c486699b849eec092b1",
            "ood": "be2ebbbf85b27558e408d5493055846ccb6a90922a490d855f283af306e42d87",
        },
    },
}


def _source_file(config: str, split: str, expected: str, cache: Path) -> Path:
    destination = cache / config / f"{split}.jsonl.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not destination.is_symlink():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
            return destination
    url = (f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/"
           f"raw/{config}/{split}.jsonl.gz")
    temporary = destination.with_suffix(".download")
    digest = hashlib.sha256()
    try:
        with urlopen(url, timeout=180) as response, temporary.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 16 * 1024 * 1024:
                    raise ValueError("benchmark source file exceeds pinned size limit")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("benchmark source hash mismatch")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _render_email(state: dict) -> str:
    email = state.get("email")
    if not isinstance(email, dict):
        raise ValueError("mailroom state lacks email")
    sender = email.get("from")
    if not isinstance(sender, dict):
        raise ValueError("mailroom state lacks sender")
    fields = [email.get("subject"), sender.get("display_name"), sender.get("email"),
              email.get("date"), email.get("body")]
    if any(not isinstance(value, str) or not value for value in fields):
        raise ValueError("mailroom state has invalid text fields")
    subject, sender_name, sender_address, date, body = fields
    return f"Subject: {subject}\nFrom: {sender_name} <{sender_address}>\nDate: {date}\n\n{body}"


def _project(record: dict, preset: dict) -> dict | None:
    head = preset["head"]
    if not isinstance(record, dict) or not str(record.get("id", "")).endswith(f":{head}"):
        return None
    if preset["config"] == "release-v2-redistributable" and record.get("source") != "customer-control-v1":
        return None
    metadata = record.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("language") != "en":
        return None
    if record.get("kind") != "choice":
        raise ValueError("preset head changed decision type")
    options, target = record.get("options"), record.get("target")
    if not isinstance(options, list) or not isinstance(target, list) or len(options) != len(target):
        raise ValueError("preset head has malformed options or target")
    labels = [option.partition(":")[0] for option in options if isinstance(option, str)]
    if len(labels) != len(options) or set(labels) != set(preset["labels"]):
        raise ValueError("preset head changed its label set")
    # The fixed-label MVP represents only hard single-label targets. Preserve
    # the exclusion count; do not turn soft distributions into arbitrary labels.
    if target.count(1.0) != 1 or any(value not in (0, 0.0, 1, 1.0) for value in target):
        return None
    state = record.get("state")
    text = _render_email(state) if head == "kind" else state
    if not isinstance(text, str) or not text.strip():
        raise ValueError("preset head has invalid state text")
    row_id, group_id = record.get("id"), record.get("group_id")
    if not isinstance(row_id, str) or not isinstance(group_id, str) or not group_id:
        raise ValueError("preset head lacks identity or group")
    return {"id": row_id, "text": text, "label": labels[target.index(1.0)], "group_id": group_id}


def prepare_preset(name: str, cache: Path) -> dict:
    if name not in PRESETS:
        raise ValueError("unknown benchmark preset")
    preset = PRESETS[name]
    source_rows: dict[str, list[dict]] = {}
    excluded: dict[str, int] = {}
    for split in SPLITS:
        path = _source_file(preset["config"], split, preset["hashes"][split], cache)
        selected = []
        candidate_count = 0
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if (isinstance(record, dict) and str(record.get("id", "")).endswith(f":{preset['head']}")
                        and (preset["config"] != "release-v2-redistributable" or record.get("source") == "customer-control-v1")
                        and isinstance(record.get("metadata"), dict) and record["metadata"].get("language") == "en"):
                    candidate_count += 1
                    row = _project(record, preset)
                    if row:
                        selected.append(row)
        source_rows[split] = selected
        excluded[split] = candidate_count - len(selected)
    validation_groups = sorted({row["group_id"] for row in source_rows["validation"]},
                               key=lambda group: hashlib.sha256(group.encode()).hexdigest())
    if len(validation_groups) < 2:
        raise ValueError("benchmark validation has too few groups")
    development_groups = set(validation_groups[:len(validation_groups) // 2])
    assigned = {
        "train": source_rows["train"],
        "development": [row for row in source_rows["validation"] if row["group_id"] in development_groups],
        "calibration": source_rows["calibration"],
        "policy": [row for row in source_rows["validation"] if row["group_id"] not in development_groups],
        "test": source_rows["test"],
    }
    records = [row for part in PARTITIONS for row in assigned[part]]
    total = len(records)
    split = {
        "schema_version": "1.0", "dataset_sha256": dataset_digest(records),
        "strategy": "upstream_splits_validation_group_halves_v1", "seed": None,
        "fractions": {part: len(assigned[part]) / total for part in PARTITIONS},
        "memberships": {part: [row["id"] for row in assigned[part]] for part in PARTITIONS},
        "group_counts": {part: len({row["group_id"] for row in assigned[part]}) for part in PARTITIONS},
        "class_counts": {part: dict(Counter(row["label"] for row in assigned[part])) for part in PARTITIONS},
        "source_dataset": DATASET_REPO, "source_revision": DATASET_REVISION,
        "source_config": preset["config"], "projection": name,
        "soft_or_unlabeled_targets_excluded": excluded,
    }
    validate_split(records, split)
    if set(row["label"] for row in assigned["train"]) != set(preset["labels"]):
        raise ValueError("benchmark training split lacks a label")
    return {"name": preset["name"], "labels": preset["labels"], "license": preset["license"],
            "records": records, "split": split, "ood": source_rows["ood"],
            "source_revision": DATASET_REVISION, "source_config": preset["config"],
            "excluded": excluded}
