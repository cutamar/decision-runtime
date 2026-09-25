"""Freeze five leakage-aware partitions before fitting model components."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from decision_runtime.model import canonical_json, sha256

PARTITIONS = ("train", "development", "calibration", "policy", "test")


def load_records(path: Path) -> list[dict[str, str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [item.get("id") for item in records]
    if not records or any(not isinstance(item, dict) or not isinstance(item.get("text"), str) or not isinstance(item.get("label"), str) for item in records):
        raise ValueError("clean dataset must contain text and label records")
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("clean dataset needs unique nonempty row IDs")
    return records


def dataset_digest(records: list[dict[str, str]]) -> str:
    return sha256(canonical_json(records))


def _components(records: list[dict[str, str]]) -> list[list[int]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    by_group: dict[str, int] = {}
    by_text: dict[str, int] = {}
    for index, record in enumerate(records):
        group = record.get("group_id")
        if group:
            if group in by_group:
                union(index, by_group[group])
            else:
                by_group[group] = index
        text_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        if text_hash in by_text:
            union(index, by_text[text_hash])
        else:
            by_text[text_hash] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)
    return list(components.values())


def make_split(
    records: list[dict[str, str]], fractions: dict[str, float], *, seed: int = 0, chronological: bool = False
) -> dict[str, Any]:
    if set(fractions) != set(PARTITIONS) or any(not isinstance(value, (int, float)) or value <= 0 for value in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("five positive partition fractions must sum to one")
    groups = _components(records)
    if chronological:
        try:
            timestamps = [datetime.fromisoformat(record["occurred_at"].replace("Z", "+00:00")) for record in records]
            groups.sort(key=lambda group: (max(timestamps[index] for index in group), min(records[index]["id"] for index in group)))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("chronological splitting requires valid occurred_at on every row") from exc
    else:
        random.Random(seed).shuffle(groups)
    boundaries = []
    cumulative = 0.0
    for part in PARTITIONS[:-1]:
        cumulative += fractions[part]
        boundaries.append(cumulative * len(records))
    membership: dict[str, list[str]] = {part: [] for part in PARTITIONS}
    group_counts = {part: 0 for part in PARTITIONS}
    row_count = 0
    part_index = 0
    for group in groups:
        while part_index < len(PARTITIONS) - 1 and row_count >= boundaries[part_index]:
            part_index += 1
        part = PARTITIONS[part_index]
        membership[part].extend(records[index]["id"] for index in group)
        group_counts[part] += 1
        row_count += len(group)
    id_lookup = {record["id"]: record for record in records}
    return {
        "schema_version": "1.0",
        "dataset_sha256": dataset_digest(records),
        "strategy": "chronological_grouped" if chronological else "seeded_grouped",
        "seed": None if chronological else seed,
        "fractions": fractions,
        "memberships": membership,
        "group_counts": group_counts,
        "class_counts": {part: dict(sorted(Counter(id_lookup[row_id]["label"] for row_id in ids).items())) for part, ids in membership.items()},
        "boundary_spanning_groups_reassigned_by_latest_time": chronological,
    }


def validate_split(records: list[dict[str, str]], manifest: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    if manifest.get("dataset_sha256") != dataset_digest(records):
        raise ValueError("split manifest does not match the dataset")
    membership = manifest.get("memberships")
    if not isinstance(membership, dict) or set(membership) != set(PARTITIONS):
        raise ValueError("split manifest needs all five partitions")
    all_ids = [row_id for part in PARTITIONS for row_id in membership[part]]
    lookup = {record["id"]: record for record in records}
    if len(all_ids) != len(records) or set(all_ids) != set(lookup):
        raise ValueError("split membership is incomplete or overlapping")
    part_by_id = {row_id: part for part in PARTITIONS for row_id in membership[part]}
    if any(len({part_by_id[records[index]["id"]] for index in group}) > 1 for group in _components(records)):
        raise ValueError("duplicate or group leakage across partitions")
    return {part: [lookup[row_id] for row_id in membership[part]] for part in PARTITIONS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a grouped MVP evaluation split")
    parser.add_argument("dataset", type=Path, help="clean JSONL from decision-import")
    parser.add_argument("--fractions", type=Path, required=True, help="JSON object with five partition fractions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chronological", action="store_true")
    args = parser.parse_args()
    try:
        records = load_records(args.dataset)
        fractions = json.loads(args.fractions.read_text(encoding="utf-8"))
        manifest = make_split(records, fractions, seed=args.seed, chronological=args.chronological)
        validate_split(records, manifest)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"split failed: {exc}\n")
    print(json.dumps({"rows": {part: len(ids) for part, ids in manifest["memberships"].items()}, "groups": manifest["group_counts"]}))


if __name__ == "__main__":
    main()
