"""Offline ONNX MiniLM encoder with exact tokenizer-length enforcement."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .model import BundleError, DecisionModel, InferenceError, InvalidInputError, Prediction, _probabilities

ALLOWED_ONNX_OPERATORS = {
    "Add", "And", "Cast", "Concat", "Constant", "ConstantOfShape", "Div", "Equal", "Erf",
    "Expand", "Flatten", "Gather", "GatherND", "GreaterOrEqual", "IsNaN", "LayerNormalization",
    "MatMul", "Max", "Mul", "Pow", "Range", "ReduceMean", "Reshape", "Shape", "Slice",
    "Softmax", "Sqrt", "Squeeze", "Sub", "Transpose", "Unsqueeze", "Where",
}


class EncoderDecisionModel(DecisionModel):
    """Apply a frozen encoder, mean pooling, and an exported linear head."""

    def __init__(self, manifest: dict[str, Any], assets: dict[str, Any], root: Path):
        try:
            import numpy as np
            import onnx
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise BundleError("encoder bundles require the neural runtime dependencies") from exc
        self.manifest = manifest
        self.labels = assets["labels.json"]["ids"]
        self.preprocessing = assets["preprocessing.json"]
        self.calibration = assets["calibration.json"]
        self.policy = assets["policy.json"]
        self.qualification = assets["qualification.json"]
        self.head = assets["head.json"]
        self._validate_common()
        if self.preprocessing.get("token_counter") != "bert_wordpiece_with_special_tokens_v1" or self.preprocessing["max_tokens"] > 256:
            raise BundleError("unsupported encoder input envelope")
        if self.preprocessing.get("pooling") != "attention_mask_mean_l2_v1" or self.preprocessing.get("embedding_dim") != 384:
            raise BundleError("unsupported encoder pooling configuration")
        dimensions = self.preprocessing["embedding_dim"]
        mode = self.head.get("score_mode")
        score_count = 1 if mode == "binary_sigmoid" and len(self.labels) == 2 else len(self.labels) if mode == "multinomial_softmax" and len(self.labels) > 2 else 0
        coefficients = self.head.get("coefficients")
        intercepts = self.head.get("intercepts")
        if not score_count or not isinstance(coefficients, list) or len(coefficients) != score_count or not isinstance(intercepts, list) or len(intercepts) != score_count:
            raise BundleError("invalid encoder head shape")
        if any(not isinstance(row, list) or len(row) != dimensions for row in coefficients):
            raise BundleError("invalid encoder head width")
        if any(not isinstance(value, (float, int)) or not math.isfinite(value) for row in coefficients for value in row) or any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in intercepts):
            raise BundleError("nonfinite encoder head values")
        self._np = np
        try:
            graph = onnx.load(str(root / "model.onnx"), load_external_data=False)
            if any(initializer.data_location == onnx.TensorProto.EXTERNAL for initializer in graph.graph.initializer):
                raise BundleError("external ONNX weights are unsupported")
            if any(item.domain not in ("", "ai.onnx") or item.version not in (14, 17, 18) for item in graph.opset_import):
                raise BundleError("unsupported ONNX opset")
            if any(node.domain not in ("", "ai.onnx") or node.op_type not in ALLOWED_ONNX_OPERATORS for node in graph.graph.node):
                raise BundleError("unsupported ONNX operator")
            onnx.checker.check_model(graph)
            self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
            self.tokenizer.no_truncation()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            self.session = ort.InferenceSession(str(root / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        except BundleError:
            raise
        except Exception as exc:
            raise BundleError("invalid encoder or tokenizer asset") from exc
        self.input_names = {value.name for value in self.session.get_inputs()}
        if self.input_names not in ({"input_ids", "attention_mask"}, {"input_ids", "attention_mask", "token_type_ids"}):
            raise BundleError("unsupported ONNX encoder inputs")
        outputs = self.session.get_outputs()
        if not outputs or len(outputs[0].shape) != 3:
            raise BundleError("unsupported ONNX encoder output")
        self._coefficient_matrix = np.asarray(coefficients, dtype=np.float64)
        self._intercepts = np.asarray(intercepts, dtype=np.float64)

    def encode(self, text: str):
        """Return a normalized embedding, or a reason to defer this input."""
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("text must be a nonempty string")
        try:
            byte_length = len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise InvalidInputError("text must be valid UTF-8") from exc
        if byte_length > self.preprocessing["max_input_bytes"]:
            return None, "input_too_long"
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        if len(encoded.ids) > self.preprocessing["max_tokens"]:
            return None, "input_too_long"
        if len(encoded.ids) <= 2:
            return None, "unsupported_input"
        np = self._np
        inputs = {
            "input_ids": np.asarray([encoded.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.asarray([encoded.type_ids], dtype=np.int64)
        try:
            values = self.session.run(None, inputs)[0]
        except Exception as exc:
            raise InferenceError("ONNX encoder inference failed") from exc
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] != len(encoded.ids) or values.shape[2] != self.preprocessing["embedding_dim"]:
            raise InferenceError("invalid ONNX embedding shape")
        mask = inputs["attention_mask"][0].astype(np.float64)
        pooled = (values[0].astype(np.float64) * mask[:, None]).sum(axis=0) / mask.sum()
        norm = float(np.linalg.norm(pooled))
        if not math.isfinite(norm) or norm <= 0:
            raise InferenceError("nonfinite or empty encoder embedding")
        return pooled / norm, None

    def predict(self, text: str, *, evaluation: bool = False) -> Prediction:
        embedding, reason = self.encode(text)
        if reason:
            return self._empty_abstention(reason, evaluation)
        scores = (self._coefficient_matrix @ embedding + self._intercepts).tolist()
        probabilities = _probabilities(scores, self.head["score_mode"], self.calibration["temperature"])
        return self._decide(probabilities, evaluation)
