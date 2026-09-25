import unittest

from decision_lab.open_jev import PRESETS, _project


class OpenJevProjectionTests(unittest.TestCase):
    def test_mailroom_projection_uses_visible_email_only(self):
        record = {
            "id": "mailroom:demo:en:invoice:base:kind", "group_id": "family-1", "kind": "choice",
            "metadata": {"language": "en", "secret_gold_hint": "should never enter model text"},
            "options": [f"{label}: definition" for label in PRESETS["mail_kind_en"]["labels"]],
            "target": [0, 1, 0, 0, 0, 0],
            "state": {"email": {"subject": "Your invoice", "from": {"display_name": "Demo", "email": "demo@example.test"},
                                "date": "2026-09-25", "body": "Payment is due."}},
        }
        row = _project(record, PRESETS["mail_kind_en"])
        self.assertEqual(row["label"], "invoice")
        self.assertIn("Your invoice", row["text"])
        self.assertNotIn("secret_gold_hint", row["text"])
        record["target"] = [0.5, 0.5, 0, 0, 0, 0]
        self.assertIsNone(_project(record, PRESETS["mail_kind_en"]))

    def test_support_projection_rejects_other_task_and_keeps_hard_label(self):
        record = {
            "id": "customer-control-v1:demo:category", "group_id": "support-demo", "kind": "choice",
            "source": "customer-control-v1", "metadata": {"language": "en"},
            "options": ["billing: description", "account: description", "feature_request: description", "bug_report: description"],
            "target": [1, 0, 0, 0], "state": "Please refund my invoice.",
        }
        row = _project(record, PRESETS["support_routing_en"])
        self.assertEqual(row["label"], "billing")
        self.assertEqual(row["text"], record["state"])
        record["source"] = "other-source"
        self.assertIsNone(_project(record, PRESETS["support_routing_en"]))


if __name__ == "__main__":
    unittest.main()
