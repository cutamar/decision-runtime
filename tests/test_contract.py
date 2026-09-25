import json
import tempfile
import unittest
from pathlib import Path

from decision_runtime import BundleError, DecisionModel
from decision_runtime.evaluate import evaluate
from decision_runtime.inspect import inspect_bundle
from decision_runtime.model import PAYLOAD_FILES, RELEASE_FILES, canonical_json, payload_digest, sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def make_bundle(root):
    root.mkdir()
    (root / "licenses").mkdir(parents=True)
    write_json(root / "linear.json", {"score_mode": "binary_sigmoid", "coefficients": [[-4.0, 4.0]], "intercepts": [0.0]})
    write_json(root / "vectorizer.json", {"token_pattern": r"(?u)\b\w\w+\b", "lowercase": True,
                                           "norm": "l2", "vocabulary": {"account": 0, "billing": 1}, "idf": [1.0, 1.0]})
    write_json(root / "labels.json", {"ids": ["account", "billing"]})
    write_json(root / "preprocessing.json", {"token_counter": "word_regex_v1", "max_tokens": 20, "max_input_bytes": 1024})
    write_json(root / "calibration.json", {"temperature": 1.0, "status": "fitted"})
    write_json(root / "policy.json", {"thresholds": {"account": 0.7, "billing": 0.7}, "review_only": [], "tie_tolerance": 1e-12})
    payload = payload_digest({name: sha256((root / name).read_bytes()) for name in PAYLOAD_FILES})
    write_json(root / "evaluation.json", {"status": "qualified"})
    write_json(root / "qualification.json", {"status": "qualified", "model_version": "fixture-v1",
                                               "inference_payload_sha256": payload,
                                               "evaluation_sha256": sha256((root / "evaluation.json").read_bytes())})
    write_json(root / "lineage.json", {"created_for": "runtime contract test"})
    (root / "model-card.md").write_text("Synthetic fixture\n", encoding="utf-8")
    (root / "licenses/NOTICE.txt").write_text("Synthetic fixture\n", encoding="utf-8")
    write_json(root / "manifest.json", {"format_version": "1.0", "artifact_kind": "sparse_linear",
                                        "model_version": "fixture-v1", "policy_version": "policy-v1",
                                        "trust": "local_unsigned",
                                        "files": {name: sha256((root / name).read_bytes()) for name in RELEASE_FILES},
                                        "inference_payload_sha256": payload})


class RuntimeContractTests(unittest.TestCase):
    def test_inspection_evaluation_and_abstention(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "bundle"
            make_bundle(root)
            with self.assertRaises(BundleError):
                DecisionModel.load(root)
            model = DecisionModel.load(root, allow_unsigned=True)
            self.assertEqual(model.predict("billing").choice, "billing")
            self.assertEqual(model.predict("account").choice, "account")
            self.assertEqual(model.predict("unknown").abstention_reason, "unsupported_input")
            self.assertEqual(model.predict("billing", evaluation=True).status, "would_accept")
            self.assertEqual(model.predict("billing", evaluation=True).choice, None)
            self.assertEqual(inspect_bundle(root)["qualification_status"], "qualified")
            labeled = Path(temp) / "labeled.jsonl"
            labeled.write_text('\n'.join(json.dumps(row) for row in [
                {"id": "1", "text": "billing", "label": "billing"},
                {"id": "2", "text": "account", "label": "account"},
                {"id": "3", "text": "unknown", "label": "account"},
            ]) + '\n', encoding="utf-8")
            report = evaluate(root, labeled)
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["supported"], 2)
            self.assertEqual(report["accepted"], 2)
            self.assertEqual(report["supported_accuracy"], 1.0)
            self.assertEqual(report["coverage"], 2 / 3)
            self.assertEqual(report["accepted_error_rate"], 0.0)
            self.assertEqual(report["abstention_reasons"], {"unsupported_input": 1})
            (root / "policy.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(BundleError):
                DecisionModel.load(root, allow_unsigned=True)


if __name__ == "__main__":
    unittest.main()
