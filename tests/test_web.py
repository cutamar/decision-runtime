import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from decision_lab.web import LocalApp, make_handler
from decision_runtime.evaluate import evaluate
from decision_runtime.inspect import inspect_bundle


class LocalWebTests(unittest.TestCase):
    def test_local_http_serves_presets_and_requires_request_token(self):
        with tempfile.TemporaryDirectory() as temp:
            app = LocalApp(Path(temp) / "runs", Path(temp) / "model")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(Request(base + "/api/presets", headers={"Host": f"127.0.0.1:{server.server_port}"})) as response:
                    self.assertIn(b"mail_kind_en", response.read())
                with urlopen(Request(base + "/", headers={"Host": f"127.0.0.1:{server.server_port}"})) as response:
                    self.assertIn(app.token.encode(), response.read())
                request = Request(base + "/api/preview", data=b"{}", method="POST",
                                  headers={"Host": f"127.0.0.1:{server.server_port}",
                                           "Content-Type": "application/json"})
                with self.assertRaises(HTTPError) as denied:
                    urlopen(request)
                self.assertEqual(denied.exception.code, 403)
                denied.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)

    def test_sparse_web_run_evaluates_holdout_and_predicts_current_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app = LocalApp(root / "runs", root / "unused-model")
            rows = ["id,text,label,group_id"]
            for index in range(120):
                label = "billing" if index % 2 else "account"
                text = f"invoice charged payment ticket{index}" if label == "billing" else f"password login account ticket{index}"
                rows.append(f"ticket-{index},{text},{label},group-{index}")
            job_id = app.start_training({
                "filename": "tickets.csv", "content": "\n".join(rows) + "\n",
                "labels": ["account", "billing"], "model": "sparse",
                "min_coverage": 0.4, "max_accepted_error": 0.2, "confidence": 0.95,
                "provenance": "Synthetic integration test", "license_notice": "Created for this test",
            })
            deadline = time.monotonic() + 20
            while app.job(job_id)["status"] in {"queued", "running"} and time.monotonic() < deadline:
                time.sleep(0.05)
            finished = app.job(job_id)
            self.assertEqual(finished["status"], "complete", finished.get("message"))
            result = finished["result"]
            self.assertEqual(result["holdout"]["rows"], result["partition_rows"]["test"])
            self.assertTrue(0 <= result["holdout"]["supported_accuracy"] <= 1)
            self.assertEqual(result["qualification_status"], "exploratory")
            bundle = Path(result["bundle"])
            self.assertEqual(inspect_bundle(bundle)["qualification_status"], "exploratory")
            restored = LocalApp(root / "runs", root / "unused-model")
            self.assertEqual(restored.current_run()["run_id"], job_id)
            prediction = restored.predict(job_id, {"text": "invoice payment issue"})
            self.assertEqual(prediction["suggested_choice"], "billing")
            self.assertIn(prediction["status"], {"would_accept", "would_defer"})
            with self.assertRaises(ValueError):
                restored.start_evaluation(job_id, {"filename": "new.csv", "content": "text,label\nx,y"})
            external = root / "new.csv"
            external.write_text("text,label\ninvoice payment issue,billing\npassword account issue,account\n", encoding="utf-8")
            report = evaluate(bundle, external)
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["supported"], 2)
            self.assertEqual(sum(sum(row) for row in report["confusion_matrix"]), 2)
            self.assertTrue(0 <= report["supported_accuracy"] <= 1)
            timing_id = app.start_benchmark({}, run_id=job_id)
            while app.job(timing_id)["status"] in {"queued", "running"} and time.monotonic() < deadline:
                time.sleep(0.05)
            timing = app.job(timing_id)
            self.assertEqual(timing["status"], "complete", timing.get("message"))
            self.assertEqual(timing["result"]["timed_calls"], 200)
            self.assertEqual(len(timing["result"]["workload_sha256"]), 64)

    def test_invalid_labels_are_rejected_before_job(self):
        with tempfile.TemporaryDirectory() as temp:
            app = LocalApp(Path(temp) / "runs", Path(temp) / "model")
            default, _ = app._settings({"min_coverage": 0.4, "max_accepted_error": 0.2,
                                        "confidence": 0.95})
            self.assertEqual(default, "adapted")
            with self.assertRaises(ValueError):
                app.start_training({"filename": "x.csv", "content": "text,label\na,A\nb,B\n",
                                    "labels": ["A", "C"], "model": "sparse",
                                    "min_coverage": 0.4, "max_accepted_error": 0.2,
                                    "confidence": 0.95})
            self.assertFalse(app.jobs)


if __name__ == "__main__":
    unittest.main()
