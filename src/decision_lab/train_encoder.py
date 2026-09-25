"""Train and evaluate the pinned frozen MiniLM ONNX encoder plus linear head."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import scipy
import sklearn
import tokenizers
from sklearn.linear_model import LogisticRegression

from decision_runtime import DecisionModel
from decision_runtime.model import ENCODER_PAYLOAD_FILES, ENCODER_RELEASE_FILES, canonical_json, payload_digest, sha256, _probabilities
from .model_source import candidate_source_info
from .splits import PARTITIONS, load_records, validate_split
from .train import _choose_threshold, _quality_report, _scores, _temperature, _write_json


def _embed_rows(rows: list[dict[str, str]], tokenizer: tokenizers.Tokenizer,
                session: ort.InferenceSession, max_tokens: int, *, batch_size: int = 16) -> tuple[list[dict[str, str]], np.ndarray]:
    supported: list[dict[str, str]] = []
    vectors: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        encoded = tokenizer.encode_batch([row["text"] for row in batch], add_special_tokens=True)
        selected = [(row, item) for row, item in zip(batch, encoded)
                    if len(row["text"].encode("utf-8")) <= 1024 * 1024 and 2 < len(item.ids) <= max_tokens]
        if not selected:
            continue
        length = max(len(item.ids) for _, item in selected)
        ids = np.zeros((len(selected), length), dtype=np.int64)
        masks = np.zeros_like(ids)
        types = np.zeros_like(ids)
        for index, (_, item) in enumerate(selected):
            size = len(item.ids)
            ids[index, :size] = item.ids
            masks[index, :size] = item.attention_mask
            types[index, :size] = item.type_ids
        outputs = session.run(None, {"input_ids": ids, "attention_mask": masks, "token_type_ids": types})[0]
        if outputs.shape != (len(selected), length, 384):
            raise ValueError("source encoder returned an unexpected shape")
        pooled = (outputs.astype(np.float64) * masks[:, :, None]).sum(axis=1) / masks.sum(axis=1)[:, None]
        norms = np.linalg.norm(pooled, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise ValueError("source encoder returned invalid embeddings")
        supported.extend(row for row, _ in selected)
        vectors.extend(pooled / norms[:, None])
    return supported, np.asarray(vectors, dtype=np.float64).reshape((-1, 384))


def _manifest(output: Path, model_version: str, policy_version: str) -> None:
    hashes = {name: sha256((output / name).read_bytes()) for name in ENCODER_RELEASE_FILES}
    _write_json(output / "manifest.json", {
        "format_version": "1.0", "artifact_kind": "encoder_onnx",
        "model_version": model_version, "policy_version": policy_version,
        "trust": "local_unsigned", "files": hashes,
        "inference_payload_sha256": payload_digest(hashes),
    })


def _train_into_bundle(dataset_path: Path, split_path: Path, plan_path: Path,
                       source: Path, output: Path, *, candidate_only: bool = False) -> dict[str, Any]:
    source_info = candidate_source_info(source)
    records = load_records(dataset_path)
    split_raw = split_path.read_bytes()
    plan_raw = plan_path.read_bytes()
    split = json.loads(split_raw)
    plan = json.loads(plan_raw)
    partitions = validate_split(records, split)
    labels = plan.get("labels")
    if not isinstance(labels, list) or not 2 <= len(labels) <= 20 or len(set(labels)) != len(labels):
        raise ValueError("plan needs 2–20 unique label IDs")
    if set(record["label"] for record in records) - set(labels):
        raise ValueError("dataset contains labels outside the plan")
    quality = plan.get("quality", {})
    min_coverage = quality.get("min_end_to_end_coverage")
    max_error = quality.get("max_accepted_error_upper_bound")
    confidence = quality.get("bound_confidence_level")
    if any(not isinstance(value, (int, float)) or not 0 < value < 1 for value in (min_coverage, max_error, confidence)):
        raise ValueError("plan needs coverage, error bound, and confidence values between zero and one")
    model_version = plan.get("model_version")
    policy_version = plan.get("policy_version")
    max_tokens = plan.get("max_tokens", 256)
    if not isinstance(model_version, str) or not model_version or not isinstance(policy_version, str) or not policy_version or not isinstance(max_tokens, int) or not 3 <= max_tokens <= 256:
        raise ValueError("plan needs model/policy versions and a 3–256 token limit")
    if any(not partitions[part] for part in PARTITIONS):
        raise ValueError("all five partitions must be nonempty")

    tokenizer = tokenizers.Tokenizer.from_file(str(source / "tokenizer.json"))
    tokenizer.no_truncation()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(source / "onnx/model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    training_rows, training_vectors = _embed_rows(partitions["train"], tokenizer, session, max_tokens)
    if {row["label"] for row in training_rows} != set(labels):
        raise ValueError("supported training inputs do not cover every label")
    classifier = LogisticRegression(max_iter=1000, random_state=0)
    classifier.fit(training_vectors, [row["label"] for row in training_rows])
    ordered_labels = [str(label) for label in classifier.classes_]
    if ordered_labels != sorted(labels):
        raise ValueError("classifier label order does not match plan")
    mode = "binary_sigmoid" if len(labels) == 2 else "multinomial_softmax"

    calibration_rows, calibration_vectors = _embed_rows(partitions["calibration"], tokenizer, session, max_tokens)
    if not calibration_rows:
        raise ValueError("calibration partition has no supported inputs")
    calibration_scores = _scores(classifier, calibration_vectors)
    temperature, calibration_status = _temperature(calibration_scores, [ordered_labels.index(row["label"]) for row in calibration_rows], mode, len(labels))
    policy_rows, policy_vectors = _embed_rows(partitions["policy"], tokenizer, session, max_tokens)
    if not policy_rows:
        raise ValueError("policy partition has no supported inputs")
    policy_scores = _scores(classifier, policy_vectors)
    policy_probabilities = [_probabilities(row, mode, temperature) for row in policy_scores]
    threshold = _choose_threshold(policy_probabilities, [ordered_labels.index(row["label"]) for row in policy_rows], max_error, confidence)

    output.mkdir(parents=True)
    (output / "licenses").mkdir()
    shutil.copyfile(source / "onnx/model.onnx", output / "model.onnx")
    shutil.copyfile(source / "tokenizer.json", output / "tokenizer.json")
    license_text = Path(__file__).resolve().parent / "licenses/Apache-2.0.txt"
    shutil.copyfile(license_text, output / "licenses/Apache-2.0.txt")
    _write_json(output / "labels.json", {"ids": ordered_labels})
    _write_json(output / "head.json", {"score_mode": mode, "coefficients": classifier.coef_.tolist(), "intercepts": classifier.intercept_.tolist()})
    _write_json(output / "preprocessing.json", {"token_counter": "bert_wordpiece_with_special_tokens_v1", "max_tokens": max_tokens,
                                                "max_input_bytes": 1024 * 1024, "pooling": "attention_mask_mean_l2_v1", "embedding_dim": 384})
    _write_json(output / "calibration.json", {"method": "temperature_scaling", "temperature": temperature, "status": calibration_status,
                                              "calibration_split_sha256": sha256(canonical_json(calibration_rows))})
    _write_json(output / "policy.json", {"thresholds": {label: threshold for label in ordered_labels},
                                         "review_only": plan.get("review_only", []), "tie_tolerance": 1e-12, "fallback": "customer_application"})
    payload_hash = payload_digest({name: sha256((output / name).read_bytes()) for name in ENCODER_PAYLOAD_FILES})
    _write_json(output / "evaluation.json", {"status": "exploratory"})
    _write_json(output / "qualification.json", {"status": "exploratory", "model_version": model_version,
                                                "inference_payload_sha256": payload_hash,
                                                "evaluation_sha256": sha256((output / "evaluation.json").read_bytes())})
    graph = onnx.load(str(source / "onnx/model.onnx"), load_external_data=False)
    _write_json(output / "lineage.json", {
        "dataset_sha256": split["dataset_sha256"], "split_manifest_sha256": sha256(split_raw),
        "evaluation_plan_sha256": sha256(plan_raw), "learner": source_info["recipe"] + "_linear_head",
        "model_source": source_info, "onnx_opsets": {item.domain or "ai.onnx": item.version for item in graph.opset_import},
        "training_seed": 0, "versions": {"python": platform.python_version(), "onnxruntime": ort.__version__,
                                          "onnx": onnx.__version__, "tokenizers": tokenizers.__version__,
                                          "scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "numpy": np.__version__},
    })
    (output / "model-card.md").write_text(
        f"# {model_version}\n\nEnglish MiniLM encoder plus fitted linear head for short-text routing.\n\n"
        f"Encoder recipe: {source_info['recipe']}. Base model: {source_info['base_model_id']} at {source_info['base_model_revision']}. Mean pooled and L2 normalized.\n\n"
        f"Data provenance: {plan.get('data_provenance_summary', 'Not supplied.')}\n\n"
        "Candidate exports default to abstention until final evaluation and signing. See evaluation.json for limitations.\n",
        encoding="utf-8",
    )
    (output / "licenses/NOTICE.txt").write_text(
        f"Base model: {source_info['base_model_id']} at {source_info['base_model_revision']}, Apache-2.0. License text: licenses/Apache-2.0.txt.\n"
        f"Dataset notice: {plan.get('data_license_notice', 'Not supplied.')}\n",
        encoding="utf-8",
    )
    _manifest(output, model_version, policy_version)
    runtime = DecisionModel.load(output, allow_unsigned=True)

    # Confirm the exact exported runtime before touching final test.
    development_rows, development_vectors = _embed_rows(partitions["development"], tokenizer, session, max_tokens)
    reference_scores = _scores(classifier, development_vectors)
    for row, scores in zip(development_rows[:64], reference_scores[:64]):
        reference = _probabilities(scores, mode, temperature)
        exported = runtime.predict(row["text"], evaluation=True).probabilities
        if exported is None or max(abs(exported[label] - value) for label, value in zip(ordered_labels, reference)) > 1e-5:
            raise ValueError("exported encoder runtime disagrees with training implementation")
    development_report, _ = _quality_report(runtime, partitions["development"], ordered_labels, confidence)
    policy_report, _ = _quality_report(runtime, partitions["policy"], ordered_labels, confidence)
    if candidate_only:
        evaluation = {
            "schema_version": "1.0", "status": "exploratory", "stage": "candidate_ready",
            "evaluated_candidates": [source_info["recipe"] + "_linear_head"],
            "dataset_sha256": split["dataset_sha256"], "split_manifest_sha256": sha256(split_raw),
            "evaluation_plan_sha256": sha256(plan_raw), "inference_payload_sha256": payload_hash,
            "partition_rows": {part: len(partitions[part]) for part in PARTITIONS},
            "partition_groups": split["group_counts"], "partition_class_counts": split["class_counts"],
            "supported_training_rows": len(training_rows), "development": development_report,
            "policy_selection": policy_report, "final_test": None, "requirements": quality,
            "evidence": {"calibration_status": calibration_status,
                         "independent_test_observations_attested": plan.get("independent_test_observations") is True},
            "limitations": ["Candidate only; final test has not been evaluated by this run."],
        }
        _write_json(output / "evaluation.json", evaluation)
        _write_json(output / "qualification.json", {"status": "exploratory", "model_version": model_version,
                                                    "inference_payload_sha256": payload_hash,
                                                    "evaluation_sha256": sha256((output / "evaluation.json").read_bytes())})
        _manifest(output, model_version, policy_version)
        DecisionModel.load(output, allow_unsigned=True)
        return evaluation
    test_groups_are_rows = split["group_counts"]["test"] == len(partitions["test"])
    independence_attested = plan.get("independent_test_observations") is True
    test_report, test_times = _quality_report(runtime, partitions["test"], ordered_labels, confidence,
                                             bound_applicable=test_groups_are_rows and independence_attested)
    evidence_ok = calibration_status == "fitted" and test_groups_are_rows and independence_attested and test_report["accepted"] > 0
    if not evidence_ok:
        status = "insufficient_evidence"
    elif test_report["end_to_end_coverage"] < min_coverage or test_report["accepted_error_upper_bound_one_sided"] > max_error:
        status = "failed"
    else:
        status = "insufficient_evidence"  # Target-machine resource gate and candidate selection remain open.
    majority_label = Counter(row["label"] for row in training_rows).most_common(1)[0][0]
    evaluation = {
        "schema_version": "1.0", "status": status, "evaluated_candidates": [source_info["recipe"] + "_linear_head"],
        "dataset_sha256": split["dataset_sha256"], "split_manifest_sha256": sha256(split_raw),
        "evaluation_plan_sha256": sha256(plan_raw), "inference_payload_sha256": payload_hash,
        "partition_rows": {part: len(partitions[part]) for part in PARTITIONS}, "partition_groups": split["group_counts"],
        "partition_class_counts": split["class_counts"], "supported_training_rows": len(training_rows),
        "majority_baseline": {"label": majority_label,
                              "test_accuracy": sum(row["label"] == majority_label for row in partitions["test"]) / len(partitions["test"])},
        "development": development_report, "final_test": test_report, "requirements": quality,
        "observed_quality_gates": {"coverage_pass": test_report["end_to_end_coverage"] >= min_coverage,
                                   "accepted_error_bound_pass": test_report["accepted_error_upper_bound_one_sided"] <= max_error if test_report["accepted_error_upper_bound_one_sided"] is not None else None},
        "evidence": {"independent_test_observations_attested": independence_attested,
                     "test_groups_equal_rows": test_groups_are_rows, "calibration_status": calibration_status},
        "local_cpu_diagnostics": {"machine": platform.platform(), "python": platform.python_version(),
                                  "warm_p50_ms": statistics.median(test_times), "warm_p95_ms": float(np.percentile(test_times, 95)),
                                  "sample_count": len(test_times), "bundle_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file())},
        "limitations": ["Synthetic or unreviewed data cannot qualify a customer workflow.",
                        "Target-machine resource gates and bounded candidate selection are not complete.",
                        "Row-level error bounds require representative, independent observations."],
    }
    _write_json(output / "evaluation.json", evaluation)
    _write_json(output / "qualification.json", {"status": status, "model_version": model_version,
                                                "inference_payload_sha256": payload_hash,
                                                "evaluation_sha256": sha256((output / "evaluation.json").read_bytes())})
    _manifest(output, model_version, policy_version)
    DecisionModel.load(output, allow_unsigned=True)
    return evaluation


def train_encoder_bundle(dataset_path: Path, split_path: Path, plan_path: Path,
                         source: Path, output: Path, *, candidate_only: bool = False) -> dict[str, Any]:
    if output.exists():
        raise ValueError("output bundle path already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        working = Path(temporary) / "bundle"
        evaluation = _train_into_bundle(dataset_path, split_path, plan_path, source, working,
                                        candidate_only=candidate_only)
        if output.exists():
            raise ValueError("output bundle path appeared during training")
        working.rename(output)
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen MiniLM ONNX routing candidate")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-only", action="store_true", help="freeze a candidate without evaluating final test")
    args = parser.parse_args()
    try:
        report = train_encoder_bundle(args.dataset, args.split, args.plan, args.source, args.output,
                                      candidate_only=args.candidate_only)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"encoder training failed: {exc}\n")
    print(json.dumps({"status": report["status"], "bundle": str(args.output),
                      "accepted": report["final_test"]["accepted"] if report["final_test"] else None}))


if __name__ == "__main__":
    main()
