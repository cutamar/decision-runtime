"""Sparse and ONNX encoder bundle loader and decision kernel.

Unsigned local bundles require an explicit trust choice. Signed bundles require
an Ed25519 public key obtained independently of the bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
PAYLOAD_FILES = (
    "linear.json", "vectorizer.json", "labels.json", "preprocessing.json", "calibration.json", "policy.json"
)
ENCODER_PAYLOAD_FILES = (
    "model.onnx", "tokenizer.json", "head.json", "labels.json", "preprocessing.json", "calibration.json", "policy.json"
)
COMMON_RELEASE_FILES = ("qualification.json", "evaluation.json", "lineage.json", "model-card.md", "licenses/NOTICE.txt")
RELEASE_FILES = PAYLOAD_FILES + COMMON_RELEASE_FILES
ENCODER_RELEASE_FILES = ENCODER_PAYLOAD_FILES + COMMON_RELEASE_FILES + ("licenses/Apache-2.0.txt",)
MAX_BUNDLE_FILE_BYTES = 64 * 1024 * 1024
MAX_ONNX_BYTES = 256 * 1024 * 1024


class InvalidInputError(ValueError):
    """The caller supplied an invalid request."""


class BundleError(ValueError):
    """The local bundle is malformed, unsupported, or untrusted."""


class InferenceError(RuntimeError):
    """The model returned an invalid numerical result."""


@dataclass(frozen=True)
class Prediction:
    schema_version: str
    status: str
    choice: str | None
    suggested_choice: str | None
    top_probability: float | None
    probabilities: dict[str, float] | None
    abstention_reason: str | None
    model_version: str
    policy_version: str
    calibration_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def payload_digest(file_hashes: dict[str, str]) -> str:
    payload_files = ENCODER_PAYLOAD_FILES if "model.onnx" in file_hashes else PAYLOAD_FILES
    return sha256(canonical_json({name: file_hashes[name] for name in payload_files}))


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BUNDLE_FILE_BYTES:
        raise BundleError(f"missing, linked, or oversized bundle file: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid JSON file: {path.name}") from exc


def _probabilities(scores: list[float], mode: str, temperature: float) -> list[float]:
    if not scores or any(not math.isfinite(value) for value in scores):
        raise InferenceError("nonfinite or empty model scores")
    if not math.isfinite(temperature) or temperature <= 0:
        raise InferenceError("invalid calibration temperature")
    if mode == "binary_sigmoid":
        if len(scores) != 1:
            raise InferenceError("binary score shape mismatch")
        scaled = scores[0] / temperature
        positive = 1 / (1 + math.exp(-scaled)) if scaled >= 0 else math.exp(scaled) / (1 + math.exp(scaled))
        return [1 - positive, positive]
    if mode != "multinomial_softmax":
        raise InferenceError("unsupported score mode")
    scaled = [value / temperature for value in scores]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    return [value / total for value in weights]


class DecisionModel:
    """Load a fixed-label bundle and apply its frozen decision policy."""

    def __init__(self, manifest: dict[str, Any], assets: dict[str, Any]):
        self.manifest = manifest
        self.labels = assets["labels.json"]["ids"]
        self.vectorizer = assets["vectorizer.json"]
        self.linear = assets["linear.json"]
        self.preprocessing = assets["preprocessing.json"]
        self.calibration = assets["calibration.json"]
        self.policy = assets["policy.json"]
        self.qualification = assets["qualification.json"]
        self._validate_assets()

    @classmethod
    def load(cls, directory: str | Path, *, allow_unsigned: bool = False,
             trusted_public_key: bytes | None = None) -> "DecisionModel":
        root = Path(directory)
        if root.is_symlink() or not root.is_dir():
            raise BundleError("bundle path must be a real directory")
        manifest_path = root / "manifest.json"
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("format_version") != "1.0":
            raise BundleError("unsupported bundle format")
        artifact_kind = manifest.get("artifact_kind")
        if artifact_kind not in {"sparse_linear", "encoder_onnx"}:
            raise BundleError("unsupported artifact kind")
        if any(not isinstance(manifest.get(key), str) or not manifest[key] for key in ("model_version", "policy_version")):
            raise BundleError("missing model or policy version")
        signature_path = root / "manifest.sig"
        if manifest.get("trust") == "signed_ed25519":
            if trusted_public_key is None:
                raise BundleError("signed bundle requires an independently trusted public key")
            if signature_path.is_symlink() or not signature_path.is_file() or signature_path.stat().st_size != 64:
                raise BundleError("missing or invalid manifest signature")
            if manifest_path.read_bytes() != canonical_json(manifest) + b"\n":
                raise BundleError("signed manifest is not canonically serialized")
            try:
                public_key = serialization.load_pem_public_key(trusted_public_key)
                if not isinstance(public_key, Ed25519PublicKey):
                    raise ValueError("expected Ed25519 public key")
                public_key.verify(signature_path.read_bytes(), canonical_json(manifest))
            except (ValueError, TypeError, InvalidSignature) as exc:
                raise BundleError("manifest signature is invalid or the public key is unsupported") from exc
        elif manifest.get("trust") == "local_unsigned":
            if signature_path.exists() or signature_path.is_symlink() or not allow_unsigned:
                raise BundleError("unsigned bundle requires explicit local trust and no signature")
        else:
            raise BundleError("unsupported bundle trust mode")
        file_hashes = manifest.get("files")
        release_files = ENCODER_RELEASE_FILES if artifact_kind == "encoder_onnx" else RELEASE_FILES
        if not isinstance(file_hashes, dict) or set(file_hashes) != set(release_files):
            raise BundleError("bundle file manifest is incomplete")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in file_hashes.values()):
            raise BundleError("invalid bundle hashes")
        assets: dict[str, Any] = {}
        for name in release_files:
            path = root / name
            limit = MAX_ONNX_BYTES if name == "model.onnx" else MAX_BUNDLE_FILE_BYTES
            if (path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                    or not path.is_file() or path.stat().st_size > limit):
                raise BundleError(f"missing, linked, or oversized bundle file: {name}")
            raw = path.read_bytes()
            if sha256(raw) != file_hashes[name]:
                raise BundleError(f"bundle file hash mismatch: {name}")
            if name.endswith(".json"):
                assets[name] = _read_json(path)
        if manifest.get("inference_payload_sha256") != payload_digest(file_hashes):
            raise BundleError("inference payload hash mismatch")
        if assets["qualification.json"].get("inference_payload_sha256") != manifest["inference_payload_sha256"]:
            raise BundleError("qualification is bound to a different payload")
        if manifest.get("model_version") != assets["qualification.json"].get("model_version"):
            raise BundleError("model version mismatch")
        if assets["qualification.json"].get("evaluation_sha256") != file_hashes["evaluation.json"]:
            raise BundleError("qualification is bound to a different evaluation")
        if assets["qualification.json"].get("status") != assets["evaluation.json"].get("status"):
            raise BundleError("qualification and evaluation statuses disagree")
        if artifact_kind == "encoder_onnx":
            from .encoder import EncoderDecisionModel
            return EncoderDecisionModel(manifest, assets, root)
        return cls(manifest, assets)

    def _validate_assets(self) -> None:
        self._validate_common()
        labels = self.labels
        vectorizer = self.vectorizer
        vocabulary = vectorizer.get("vocabulary")
        idf = vectorizer.get("idf")
        if vectorizer.get("token_pattern") != TOKEN_PATTERN.pattern or vectorizer.get("lowercase") is not True or vectorizer.get("norm") != "l2":
            raise BundleError("unsupported vectorizer preprocessing")
        if not isinstance(vocabulary, dict) or not isinstance(idf, list) or len(vocabulary) != len(idf):
            raise BundleError("invalid vectorizer shape")
        if any(not isinstance(term, str) or not isinstance(index, int) for term, index in vocabulary.items()) or set(vocabulary.values()) != set(range(len(idf))):
            raise BundleError("invalid vocabulary indexes")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in idf):
            raise BundleError("invalid IDF values")
        coefficients = self.linear.get("coefficients")
        intercepts = self.linear.get("intercepts")
        mode = self.linear.get("score_mode")
        score_count = 1 if mode == "binary_sigmoid" and len(labels) == 2 else len(labels) if mode == "multinomial_softmax" and len(labels) > 2 else 0
        if not score_count or not isinstance(coefficients, list) or len(coefficients) != score_count or not isinstance(intercepts, list) or len(intercepts) != score_count:
            raise BundleError("invalid classifier shape")
        if any(not isinstance(row, list) or len(row) != len(idf) for row in coefficients):
            raise BundleError("invalid coefficient width")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for row in coefficients for value in row) or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in intercepts):
            raise BundleError("nonfinite classifier values")
        if self.preprocessing.get("token_counter") != "word_regex_v1":
            raise BundleError("unsupported input envelope")

    def _validate_common(self) -> None:
        labels = self.labels
        if not isinstance(labels, list) or not 2 <= len(labels) <= 20 or any(not isinstance(x, str) or not x for x in labels) or len(set(labels)) != len(labels):
            raise BundleError("invalid label IDs")
        if not isinstance(self.preprocessing.get("max_tokens"), int) or self.preprocessing["max_tokens"] < 1:
            raise BundleError("invalid token limit")
        if not isinstance(self.preprocessing.get("max_input_bytes"), int) or not 1 <= self.preprocessing["max_input_bytes"] <= 1024 * 1024:
            raise BundleError("invalid byte limit")
        temperature = self.calibration.get("temperature")
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise BundleError("invalid calibration")
        if self.calibration.get("status") not in {"fitted", "unfitted"}:
            raise BundleError("invalid calibration status")
        thresholds = self.policy.get("thresholds")
        if not isinstance(thresholds, dict) or set(thresholds) != set(labels) or any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in thresholds.values()):
            raise BundleError("invalid thresholds")
        review_only = self.policy.get("review_only")
        if not isinstance(review_only, list) or not set(review_only) <= set(labels):
            raise BundleError("invalid review-only labels")
        tolerance = self.policy.get("tie_tolerance")
        if not isinstance(tolerance, (int, float)) or not 0 <= tolerance < 1:
            raise BundleError("invalid tie tolerance")
        if self.qualification.get("status") not in {"qualified", "exploratory", "failed", "insufficient_evidence"}:
            raise BundleError("invalid qualification status")

    def _empty_abstention(self, reason: str, evaluation: bool) -> Prediction:
        return Prediction("1.0", "would_defer" if evaluation else "abstain", None, None, None, None, reason,
                          self.manifest["model_version"], self.manifest["policy_version"], self.calibration["status"])

    def _vectorize(self, tokens: list[str]) -> dict[int, float]:
        counts = Counter(tokens)
        vocabulary = self.vectorizer["vocabulary"]
        idf = self.vectorizer["idf"]
        values = {vocabulary[token]: count * idf[vocabulary[token]] for token, count in counts.items() if token in vocabulary}
        norm = math.sqrt(sum(value * value for value in values.values()))
        return {index: value / norm for index, value in values.items()} if norm else {}

    def predict(self, text: str, *, evaluation: bool = False) -> Prediction:
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("text must be a nonempty string")
        try:
            byte_length = len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise InvalidInputError("text must be valid UTF-8") from exc
        if byte_length > self.preprocessing["max_input_bytes"]:
            return self._empty_abstention("input_too_long", evaluation)
        tokens = TOKEN_PATTERN.findall(text.lower())
        if len(tokens) > self.preprocessing["max_tokens"]:
            return self._empty_abstention("input_too_long", evaluation)
        if not tokens:
            return self._empty_abstention("unsupported_input", evaluation)
        vector = self._vectorize(tokens)
        if not vector:
            return self._empty_abstention("unsupported_input", evaluation)
        scores = [bias + sum(row[index] * value for index, value in vector.items())
                  for row, bias in zip(self.linear["coefficients"], self.linear["intercepts"])]
        probabilities = _probabilities(scores, self.linear["score_mode"], self.calibration["temperature"])
        return self._decide(probabilities, evaluation)

    def _decide(self, probabilities: list[float], evaluation: bool) -> Prediction:
        if len(probabilities) != len(self.labels) or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities) or abs(sum(probabilities) - 1) > 1e-9:
            raise InferenceError("invalid probability vector")
        ranking = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)
        winner = self.labels[ranking[0]]
        top = probabilities[ranking[0]]
        reason = None
        if top - probabilities[ranking[1]] <= self.policy["tie_tolerance"]:
            reason = "ambiguous_top_choice"
        elif winner in self.policy["review_only"]:
            reason = "label_review_only"
        elif top < self.policy["thresholds"][winner]:
            reason = "below_threshold"
        elif not evaluation and self.qualification["status"] != "qualified":
            reason = "unqualified_policy"
        accepted = reason is None
        status = ("would_accept" if accepted else "would_defer") if evaluation else ("accepted" if accepted else "abstain")
        return Prediction("1.0", status, winner if accepted and not evaluation else None, winner, top,
                          dict(zip(self.labels, probabilities)), reason,
                          self.manifest["model_version"], self.manifest["policy_version"], self.calibration["status"])
