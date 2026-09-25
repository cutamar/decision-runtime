import json
import unittest

from decision_lab.jevbench_public import _score, parse_submission, score_public_submission


def task(task_id, tier, kind, labels, expected):
    return {"id": task_id, "tier": tier, "question": {"type": kind},
            "labels": labels, "expected": expected, "provenance": {}}


class JevBenchPublicTests(unittest.TestCase):
    def test_missing_and_malformed_answers_keep_full_denominator(self):
        tasks = {
            "n": task("n", "original", "noul", ["no", "yes"], "yes"),
            "c": task("c", "easy", "choice", ["a", "b"], "b"),
            "s": task("s", "hard", "score", ["0", "1"], 1),
            "missing": task("missing", "hard", "choice", ["x", "y"], "y"),
        }
        rows = [
            {"task_id": "n", "probs_as_returned": {"no": 0.1, "yes": 0.9}, "correct": False},
            {"task_id": "c", "probabilities": {"a": 0.1, "b": 0.905}},
            {"task_id": "s", "probabilities": {"0": 0.2, "1": 0.3}},
        ]
        report = score_public_submission(tasks, rows)
        self.assertEqual(report["overall"]["planned"], 4)
        self.assertEqual(report["overall"]["submitted"], 3)
        self.assertEqual(report["overall"]["correct"], 2)
        self.assertEqual(report["overall"]["valid"], 2)
        self.assertEqual(report["overall"]["strict_valid"], 1)
        self.assertEqual(report["calibration_rows"], 2)
        self.assertIsNone(report["official_jevbench_score"])

    def test_label_only_answer_has_no_calibration_and_duplicates_fail(self):
        example = task("one", "original", "choice", ["a", "b"], "b")
        self.assertEqual(_score({"label": "b"}, example)["correct"], True)
        report = score_public_submission({"one": example}, [{"task_id": "one", "label": "b"}])
        self.assertEqual(report["calibration_rows"], 0)
        self.assertIsNone(report["ece_10_bins"])
        raw = (json.dumps({"task_id": "one", "label": "a"}) + "\n") * 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_submission(raw.encode())
        with self.assertRaisesRegex(ValueError, "unknown"):
            score_public_submission({"one": example}, [{"task_id": "other", "label": "b"}])


if __name__ == "__main__":
    unittest.main()
