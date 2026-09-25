"""Train and evaluate the sparse local feasibility candidate.

Only the training partition fits features and weights. Calibration and policy
selection use their own partitions. The final test is evaluated once through
the exported offline runtime after the payload has been frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from scipy.stats import beta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

from decision_runtime import DecisionModel
from decision_runtime.model import PAYLOAD_FILES, RELEASE_FILES, TOKEN_PATTERN, canonical_json, payload_digest, sha256, _probabilities
from .splits import PARTITIONS, load_records, validate_split


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _manifest(bundle: Path, model_version: str, policy_version: str) -> dict[str, Any]:
    hashes = {name: sha256((bundle / name).read_bytes()) for name in RELEASE_FILES}
    manifest = {
        "format_version": "1.0",
        "artifact_kind": "sparse_linear",
        "model_version": model_version,
        "policy_version": policy_version,
        "trust": "local_unsigned",
        "files": hashes,
        "inference_payload_sha256": payload_digest(hashes),
    }
    _write_json(bundle / "manifest.json", manifest)
    return manifest


def _scores(model: LogisticRegression, features: Any) -> list[list[float]]:
    values = model.decision_function(features)
    if len(model.classes_) == 2:
        return [[float(value)] for value in values]
    return [[float(value) for value in row] for row in values]


def _temperature(scores: list[list[float]], targets: list[int], mode: str, label_count: int) -> tuple[float, str]:
    if len(scores) < 20 or len(set(targets)) != label_count:
        return 1.0, "unfitted"

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        probabilities = [_probabilities(row, mode, temperature) for row in scores]
        return float(log_loss(targets, probabilities, labels=list(range(label_count))))

    result = minimize_scalar(objective, bounds=(math.log(0.05), math.log(20)), method="bounded")
    if not result.success or not math.isfinite(result.x):
        return 1.0, "unfitted"
    return float(math.exp(result.x)), "fitted"


def _upper_bound(errors: int, accepted: int, confidence: float) -> float | None:
    if accepted == 0:
        return None
    if errors == accepted:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, accepted - errors))


def _choose_threshold(probabilities: list[list[float]], truths: list[int], max_error: float, confidence: float) -> float:
    best = (0, 1.0)
    for threshold in [step / 100 for step in range(50, 101)]:
        accepted = [index for index, row in enumerate(probabilities) if max(row) >= threshold and sorted(row, reverse=True)[0] - sorted(row, reverse=True)[1] > 1e-12]
        errors = sum(int(np.argmax(probabilities[index]) != truths[index]) for index in accepted)
        bound = _upper_bound(errors, len(accepted), confidence)
        if bound is not None and bound <= max_error and len(accepted) > best[0]:
            best = (len(accepted), threshold)
    return best[1]


def _quality_report(
    model: DecisionModel, rows: list[dict[str, str]], labels: list[str], confidence: float,
    *, bound_applicable: bool = False,
) -> tuple[dict[str, Any], list[float]]:
    predictions = []
    times = []
    for record in rows:
        started = time.perf_counter()
        prediction = model.predict(record["text"], evaluation=True)
        times.append((time.perf_counter() - started) * 1000)
        predictions.append(prediction)
    supported = [(row, prediction) for row, prediction in zip(rows, predictions) if prediction.probabilities is not None]
    truths = [row["label"] for row, _ in supported]
    guesses = [prediction.suggested_choice for _, prediction in supported]
    distributions = [prediction.probabilities for _, prediction in supported]
    accepted = [(row, prediction) for row, prediction in zip(rows, predictions) if prediction.status == "would_accept"]
    errors = sum(row["label"] != prediction.suggested_choice for row, prediction in accepted)
    accepted_count = len(accepted)
    bound = _upper_bound(errors, accepted_count, confidence) if bound_applicable else None
    per_label_accepted = {}
    for label in labels:
        selected = [(row, prediction) for row, prediction in accepted if prediction.suggested_choice == label]
        per_label_accepted[label] = {
            "count": len(selected),
            "errors": sum(row["label"] != label for row, _ in selected),
            "precision": sum(row["label"] == label for row, _ in selected) / len(selected) if selected else None,
        }
    report: dict[str, Any] = {
        "valid_requests": len(rows),
        "supported_requests": len(supported),
        "accepted": accepted_count,
        "deferred": len(rows) - accepted_count,
        "accepted_errors": errors,
        "end_to_end_coverage": accepted_count / len(rows) if rows else None,
        "supported_coverage": accepted_count / len(supported) if supported else None,
        "accepted_error_rate": errors / accepted_count if accepted_count else None,
        "accepted_accuracy": 1 - errors / accepted_count if accepted_count else None,
        "accepted_error_upper_bound_one_sided": bound,
        "bound_confidence": confidence,
        "bound_method_status": "clopper_pearson_independent" if bound_applicable else "not_applicable_independence_unverified",
        "per_predicted_label_accepted": per_label_accepted,
        "abstention_reasons": {reason: sum(prediction.abstention_reason == reason for prediction in predictions) for reason in sorted({prediction.abstention_reason for prediction in predictions if prediction.abstention_reason})},
    }
    if supported:
        matrix = [[distribution[label] for label in labels] for distribution in distributions]
        report["classification"] = {
            "accuracy": float(accuracy_score(truths, guesses)),
            "per_label": classification_report(truths, guesses, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truths, guesses, labels=labels).tolist(),
            "negative_log_likelihood": float(log_loss(truths, matrix, labels=labels)),
            "multiclass_brier_mean_sum": float(np.mean([sum((probability - int(label == truth)) ** 2 for label, probability in zip(labels, row)) for truth, row in zip(truths, matrix)])),
        }
        bins = []
        for index in range(10):
            members = [(truth, guess, max(row)) for truth, guess, row in zip(truths, guesses, matrix) if min(int(max(row) * 10), 9) == index]
            bins.append({"lower": index / 10, "upper": (index + 1) / 10, "count": len(members),
                         "mean_confidence": statistics.mean(item[2] for item in members) if members else None,
                         "accuracy": statistics.mean(item[0] == item[1] for item in members) if members else None})
        report["calibration_bins"] = bins
        report["ece_10_bins"] = sum(item["count"] * abs(item["accuracy"] - item["mean_confidence"]) for item in bins if item["count"]) / len(supported)
    else:
        report["classification"] = None
        report["calibration_bins"] = []
        report["ece_10_bins"] = None
    return report, times


def _train_into_bundle(dataset_path: Path, split_path: Path, plan_path: Path, output: Path,
                       *, candidate_only: bool = False) -> dict[str, Any]:
    if output.exists():
        raise ValueError("output bundle path already exists")
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
    data_license_notice = plan.get("data_license_notice", "No dataset license notice supplied by the operator.")
    data_provenance_summary = plan.get("data_provenance_summary", "Dataset provenance was not described in the plan.")
    if not isinstance(model_version, str) or not model_version or not isinstance(policy_version, str) or not policy_version or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("plan needs model/policy versions and a positive max_tokens")
    if any(not isinstance(value, str) or not value.strip() or len(value) > 4000 for value in (data_license_notice, data_provenance_summary)):
        raise ValueError("dataset provenance and license notices must be short nonempty strings")
    train = partitions["train"]
    if len({row["label"] for row in train}) != len(labels):
        raise ValueError("training partition does not contain every planned label")
    if any(not partitions[part] for part in PARTITIONS[1:]):
        raise ValueError("development, calibration, policy, and test partitions must be nonempty")

    vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN.pattern, lowercase=True, max_features=20_000)
    train_features = vectorizer.fit_transform([row["text"] for row in train])
    classifier = LogisticRegression(max_iter=1000, random_state=0)
    classifier.fit(train_features, [row["label"] for row in train])
    if list(classifier.classes_) != sorted(labels):
        raise ValueError("classifier label order does not match the plan")
    ordered_labels = [str(label) for label in classifier.classes_]
    mode = "binary_sigmoid" if len(labels) == 2 else "multinomial_softmax"
    calibration = partitions["calibration"]
    calibration_scores = _scores(classifier, vectorizer.transform([row["text"] for row in calibration]))
    temperature, calibration_status = _temperature(calibration_scores, [ordered_labels.index(row["label"]) for row in calibration], mode, len(labels))
    policy_rows = partitions["policy"]
    policy_scores = _scores(classifier, vectorizer.transform([row["text"] for row in policy_rows]))
    policy_probs = [_probabilities(row, mode, temperature) for row in policy_scores]
    threshold = _choose_threshold(policy_probs, [ordered_labels.index(row["label"]) for row in policy_rows], max_error, confidence)

    output.mkdir(parents=True)
    (output / "licenses").mkdir()
    _write_json(output / "labels.json", {"ids": ordered_labels})
    _write_json(output / "vectorizer.json", {"token_pattern": TOKEN_PATTERN.pattern, "lowercase": True, "norm": "l2", "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()}, "idf": vectorizer.idf_.tolist()})
    _write_json(output / "linear.json", {"score_mode": mode, "coefficients": classifier.coef_.tolist(), "intercepts": classifier.intercept_.tolist()})
    _write_json(output / "preprocessing.json", {"token_counter": "word_regex_v1", "max_tokens": max_tokens, "max_input_bytes": 1024 * 1024})
    _write_json(output / "calibration.json", {"method": "temperature_scaling", "temperature": temperature, "status": calibration_status, "calibration_split_sha256": sha256(canonical_json(calibration))})
    _write_json(output / "policy.json", {"thresholds": {label: threshold for label in ordered_labels}, "review_only": plan.get("review_only", []), "tie_tolerance": 1e-12, "fallback": "customer_application"})
    payload_hash = payload_digest({name: sha256((output / name).read_bytes()) for name in PAYLOAD_FILES})
    _write_json(output / "evaluation.json", {"status": "exploratory"})
    _write_json(output / "qualification.json", {"status": "exploratory", "model_version": model_version, "inference_payload_sha256": payload_hash, "evaluation_sha256": sha256((output / "evaluation.json").read_bytes())})
    _write_json(output / "lineage.json", {
        "dataset_sha256": split["dataset_sha256"],
        "split_manifest_sha256": sha256(split_raw),
        "evaluation_plan_sha256": sha256(plan_raw),
        "learner": "tfidf_logistic_regression",
        "training_seed": 0,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "numpy": np.__version__},
    })
    (output / "model-card.md").write_text(
        f"# {model_version}\n\n"
        "Intended use: short English, single-label internal request routing with a stable taxonomy.\n\n"
        f"Data provenance: {data_provenance_summary}\n\n"
        "Candidate bundles begin unsigned; a final release may be signed. The runtime does not execute a business action or call a remote service.\n\n"
        "Check evaluation.json and qualification.json before use. The sparse candidate alone does not establish the full MVP comparison.\n\n"
        "Known limits: unfamiliar topics, missing decision-time facts, changing labels, long input, and distribution shift may cause errors or abstention.\n",
        encoding="utf-8",
    )
    (output / "licenses" / "NOTICE.txt").write_text(
        "This sparse bundle contains customer-specific fitted parameters and vocabulary. No pretrained model weights are included.\n"
        f"Dataset notice: {data_license_notice}\n"
        "The Python training environment uses scikit-learn, SciPy, and NumPy; inspect the pinned versions and their licenses before redistribution.\n",
        encoding="utf-8",
    )
    _manifest(output, model_version, policy_version)
    runtime = DecisionModel.load(output, allow_unsigned=True)

    # Check the serialized vectorizer and linear head before touching final test.
    development_rows = partitions["development"]
    reference_scores = _scores(classifier, vectorizer.transform([row["text"] for row in development_rows]))
    for row, scores in zip(development_rows, reference_scores):
        reference = _probabilities(scores, mode, temperature)
        exported = runtime.predict(row["text"], evaluation=True).probabilities
        if exported is None or max(abs(exported[label] - value) for label, value in zip(ordered_labels, reference)) > 1e-8:
            raise ValueError("exported runtime disagrees with training implementation on development data")

    # Development is for diagnostics; final confirmation runs only after the payload is frozen.
    development_report, _ = _quality_report(runtime, development_rows, ordered_labels, confidence)
    policy_report, _ = _quality_report(runtime, policy_rows, ordered_labels, confidence)
    if candidate_only:
        evaluation = {
            "schema_version": "1.0", "status": "exploratory", "stage": "candidate_ready",
            "evaluated_candidates": ["tfidf_logistic_regression"],
            "dataset_sha256": split["dataset_sha256"], "split_manifest_sha256": sha256(split_raw),
            "evaluation_plan_sha256": sha256(plan_raw), "inference_payload_sha256": payload_hash,
            "partition_rows": {part: len(partitions[part]) for part in PARTITIONS},
            "partition_groups": split["group_counts"], "partition_class_counts": split["class_counts"],
            "development": development_report, "policy_selection": policy_report,
            "final_test": None, "requirements": quality,
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
    test_report, test_times = _quality_report(
        runtime, partitions["test"], ordered_labels, confidence,
        bound_applicable=test_groups_are_rows and independence_attested,
    )
    evidence_ok = calibration_status == "fitted" and test_groups_are_rows and independence_attested and test_report["accepted"] > 0
    if not evidence_ok:
        status = "insufficient_evidence"
    elif test_report["end_to_end_coverage"] < min_coverage or test_report["accepted_error_upper_bound_one_sided"] > max_error:
        status = "failed"
    else:
        status = "qualified"
    majority_label = Counter(row["label"] for row in train).most_common(1)[0][0]
    # The CPU measurements are local diagnostics, not a named customer-machine gate.
    evaluation = {
        "schema_version": "1.0",
        "status": status,
        "evaluated_candidates": ["tfidf_logistic_regression"],
        "dataset_sha256": split["dataset_sha256"],
        "split_manifest_sha256": sha256(split_raw),
        "evaluation_plan_sha256": sha256(plan_raw),
        "inference_payload_sha256": payload_hash,
        "partition_rows": {part: len(partitions[part]) for part in PARTITIONS},
        "partition_groups": split["group_counts"],
        "partition_class_counts": split["class_counts"],
        "majority_baseline": {
            "label": majority_label,
            "test_accuracy": sum(row["label"] == majority_label for row in partitions["test"]) / len(partitions["test"]),
        },
        "development": development_report,
        "final_test": test_report,
        "requirements": quality,
        "observed_quality_gates": {
            "coverage_pass": test_report["end_to_end_coverage"] >= min_coverage,
            "accepted_error_bound_pass": test_report["accepted_error_upper_bound_one_sided"] <= max_error if test_report["accepted_error_upper_bound_one_sided"] is not None else None,
        },
        "evidence": {"independent_test_observations_attested": independence_attested, "test_groups_equal_rows": test_groups_are_rows, "calibration_status": calibration_status},
        "local_cpu_diagnostics": {"machine": platform.platform(), "python": platform.python_version(), "warm_p50_ms": statistics.median(test_times), "warm_p95_ms": float(np.percentile(test_times, 95)), "sample_count": len(test_times), "bundle_bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file())},
        "versions": {"scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "numpy": np.__version__},
        "limitations": ["Sparse candidate only; encoder candidates and target-machine resource gates are not yet implemented.", "Row-level binomial bound needs genuinely independent, representative observations even when group IDs are unique."],
    }
    # Until deployment limits are checked, this cannot be an actionable release.
    if evaluation["status"] == "qualified":
        evaluation["status"] = "insufficient_evidence"
        evaluation["limitations"].append("Target-machine latency, memory, and bundle-size gates have not been verified.")
    _write_json(output / "evaluation.json", evaluation)
    _write_json(output / "qualification.json", {"status": evaluation["status"], "model_version": model_version, "inference_payload_sha256": payload_hash, "evaluation_sha256": sha256((output / "evaluation.json").read_bytes())})
    _manifest(output, model_version, policy_version)
    DecisionModel.load(output, allow_unsigned=True)
    return evaluation


def train_bundle(dataset_path: Path, split_path: Path, plan_path: Path, output: Path,
                 *, candidate_only: bool = False) -> dict[str, Any]:
    """Build in a private temporary directory and publish only a complete bundle."""
    if output.exists():
        raise ValueError("output bundle path already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        working = Path(temporary) / "bundle"
        evaluation = _train_into_bundle(dataset_path, split_path, plan_path, working, candidate_only=candidate_only)
        if output.exists():
            raise ValueError("output bundle path appeared during training")
        working.rename(output)
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one sparse MVP candidate")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-only", action="store_true", help="freeze a candidate without evaluating final test")
    args = parser.parse_args()
    try:
        report = train_bundle(args.dataset, args.split, args.plan, args.output, candidate_only=args.candidate_only)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"training failed: {exc}\n")
    print(json.dumps({"status": report["status"], "bundle": str(args.output),
                      "final_test_accepted": report["final_test"]["accepted"] if report["final_test"] else None}))


if __name__ == "__main__":
    main()
