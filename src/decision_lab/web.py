"""Single-operator, loopback-only exploratory training and evaluation UI."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import secrets
import statistics
import threading
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from decision_runtime.benchmark import benchmark
from decision_runtime.evaluate import evaluate
from decision_runtime.model import DecisionModel, InvalidInputError, canonical_json, sha256
from .adapt_encoder import adapt_encoder
from .importer import MAX_UPLOAD_BYTES, ImportErrorDetail, _validate_labels, validate_file
from .model_source import download_source
from .open_jev import DATASET_REVISION, PRESETS, prepare_preset
from .splits import make_split, validate_split
from .train import train_bundle

FRACTIONS = {"train": 0.4, "development": 0.15, "calibration": 0.15, "policy": 0.15, "test": 0.15}
ASSETS = Path(__file__).with_name("web_assets")


def discover_labels(filename: str, raw: bytes) -> list[str]:
    if len(raw) > MAX_UPLOAD_BYTES or not raw:
        raise ValueError("file must be nonempty and at most 100 MiB")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".jsonl"}:
        raise ValueError("upload a CSV or JSONL file")
    decoded = raw.decode("utf-8-sig")
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(decoded, newline=""), strict=True)
        if not reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV needs a label column")
        labels = {row.get("label") for row in reader if isinstance(row.get("label"), str)}
    else:
        labels = set()
        for line in decoded.splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict) and isinstance(item.get("label"), str):
                    labels.add(item["label"])
    result = sorted(labels)
    _validate_labels(result)
    return result


class LocalApp:
    def __init__(self, workdir: Path, source: Path):
        self.workdir = workdir
        self.source = source
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.current_model_id: str | None = None
        self.current_model: DecisionModel | None = None
        self.token = secrets.token_hex(24)
        workdir.mkdir(mode=0o700, parents=True, exist_ok=True)
        workdir.chmod(0o700)

    def preview(self, payload: dict) -> dict:
        filename, raw = self._upload(payload)
        return {"labels": discover_labels(filename, raw), "bytes": len(raw)}

    def _upload(self, payload: dict) -> tuple[str, bytes]:
        filename = payload.get("filename")
        content = payload.get("content")
        if not isinstance(filename, str) or Path(filename).name != filename or Path(filename).suffix.lower() not in {".csv", ".jsonl"}:
            raise ValueError("filename must be a simple .csv or .jsonl name")
        if not isinstance(content, str):
            raise ValueError("file content must be UTF-8 text")
        raw = content.encode("utf-8")
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("file must be nonempty and at most 100 MiB")
        return filename, raw

    def _start(self, kind: str, task, *, run_id: str | None = None) -> str:
        with self.lock:
            if any(job["status"] in {"queued", "running"} for job in self.jobs.values()):
                raise ValueError("another local job is running; wait for it to finish")
            job_id = secrets.token_hex(8)
            self.jobs[job_id] = {"id": job_id, "kind": kind, "run_id": run_id,
                                 "status": "queued", "phase": "queued", "message": "Queued"}

        def run():
            self._update(job_id, status="running", phase="preparing", message="Starting")
            try:
                result = task(job_id)
                if kind == "train":
                    (self.workdir / job_id / "web-result.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self._update(job_id, status="complete", phase="complete", message="Complete", result=result)
            except Exception as exc:
                self._update(job_id, status="failed", phase="failed", message=str(exc)[:400])

        threading.Thread(target=run, name=f"decision-{kind}-{job_id}", daemon=True).start()
        return job_id

    def _update(self, job_id: str, **changes) -> None:
        with self.lock:
            self.jobs[job_id].update(changes)

    def job(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("unknown job")
            return dict(self.jobs[job_id])

    def run_result(self, run_id: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{16}", run_id):
            raise KeyError("unknown run")
        with self.lock:
            job = self.jobs.get(run_id)
            if job is not None and job["status"] != "complete":
                raise ValueError("model preparation is still running")
        path = self.workdir / run_id / "web-result.json"
        if path.is_symlink() or not path.is_file():
            raise KeyError("unknown run")
        result = json.loads(path.read_text(encoding="utf-8"))
        result["bundle"] = str(self.workdir / run_id / "bundle")
        result["benchmark_texts"] = str(self.workdir / run_id / "benchmark-texts.json")
        return result

    def current_run(self) -> dict | None:
        candidates = [path for path in self.workdir.glob("*/web-result.json")
                      if re.fullmatch(r"[0-9a-f]{16}", path.parent.name) and path.is_file() and not path.is_symlink()]
        if not candidates:
            return None
        latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        return self.run_result(latest.parent.name)

    def _settings(self, payload: dict) -> tuple[str, dict]:
        model_kind = payload.get("model", "adapted")
        if not isinstance(model_kind, str) or model_kind not in {"sparse", "frozen", "adapted"}:
            raise ValueError("choose sparse, frozen or adapted")
        try:
            coverage = float(payload.get("min_coverage"))
            max_error = float(payload.get("max_accepted_error"))
            confidence = float(payload.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("quality targets must be numbers") from exc
        if any(not 0 < value < 1 for value in (coverage, max_error, confidence)):
            raise ValueError("quality targets must be between zero and one")
        return model_kind, {"min_end_to_end_coverage": coverage,
                            "max_accepted_error_upper_bound": max_error,
                            "bound_confidence_level": confidence}

    def _fit(self, job_id: str, root: Path, records: list[dict], split: dict,
             labels: list[str], model_kind: str, quality: dict,
             provenance: str, license_notice: str, import_summary: dict,
             *, preset: str | None = None, ood: list[dict] | None = None) -> dict:
        self._update(job_id, phase="preparing", message="Preparing records and checking the split")
        dataset = root / "clean.jsonl"
        with dataset.open("x", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        validate_split(records, split)
        split_path = root / "split.json"
        split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        selected = {row["id"]: row for row in records}
        with (root / "test.jsonl").open("x", encoding="utf-8") as output:
            for row_id in split["memberships"]["test"]:
                output.write(json.dumps(selected[row_id], ensure_ascii=False) + "\n")
        if preset:
            with (root / "ood.jsonl").open("x", encoding="utf-8") as output:
                for row in ood or []:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
        plan = {
            "labels": labels,
            "model_version": f"local-{model_kind}-{job_id}",
            "policy_version": f"local-policy-{job_id}",
            "max_tokens": 256,
            "independent_test_observations": False,
            "data_provenance_summary": provenance,
            "data_license_notice": license_notice,
            "quality": quality,
        }
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        bundle = root / "bundle"
        if model_kind == "sparse":
            self._update(job_id, phase="fitting", message="Fitting the sparse classifier and selecting a review threshold")
            report = train_bundle(dataset, split_path, plan_path, bundle, candidate_only=True)
        else:
            self._update(job_id, phase="source", message="Downloading and verifying MiniLM files")
            download_source(self.source, include_weights=model_kind == "adapted")
            if model_kind == "adapted":
                self._update(job_id, phase="adapting", message="Adapting MiniLM's last layer")
                adapted = root / "adapted-source"
                adapt_encoder(dataset, split_path, self.source, adapted)
                source = adapted
            else:
                source = self.source
            self._update(job_id, phase="fitting", message="Fitting the classifier and checking development data")
            from .train_encoder import train_encoder_bundle
            report = train_encoder_bundle(dataset, split_path, plan_path, source, bundle, candidate_only=True)
        development_ids = set(split["memberships"]["development"])
        texts = [row["text"] for row in records if row["id"] in development_ids]
        (root / "benchmark-texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._update(job_id, phase="evaluating", message="Evaluating the current model on the holdout")
        holdout = evaluate(bundle, root / "test.jsonl")
        holdout["source_split"] = "test"
        if preset:
            holdout["benchmark_note"] = "Derived fixed-label subset of public synthetic data; not an Open-Jev or JevBench score."
        (root / "holdout-report.json").write_text(json.dumps(holdout, indent=2) + "\n", encoding="utf-8")
        return {"run_id": job_id, "model": model_kind, "bundle": str(bundle),
                "benchmark_texts": str(root / "benchmark-texts.json"),
                "preset": preset, "import": import_summary,
                "partition_rows": report["partition_rows"],
                "development": report["development"], "holdout": holdout,
                "policy_selection": report["policy_selection"],
                "qualification_status": report["status"],
                "note": "Local diagnostic run; metrics do not establish production performance."}

    def start_training(self, payload: dict) -> str:
        if payload.get("preset"):
            return self.start_preset(payload)
        filename, raw = self._upload(payload)
        labels = payload.get("labels")
        _validate_labels(labels)
        if sorted(labels) != discover_labels(filename, raw):
            raise ValueError("label IDs must exactly match the uploaded file")
        model_kind, quality = self._settings(payload)
        provenance = payload.get("provenance", "Operator-supplied exploratory data.")
        license_notice = payload.get("license_notice", "Operator-supplied data; rights not recorded.")
        if any(not isinstance(value, str) or not value.strip() or len(value) > 4000 for value in (provenance, license_notice)):
            raise ValueError("short provenance and data-rights notes are required")

        def train(job_id: str) -> dict:
            root = self.workdir / job_id
            root.mkdir(mode=0o700)
            self._update(job_id, phase="preparing", message="Validating your labeled file")
            uploaded = root / filename
            uploaded.write_bytes(raw)
            imported = validate_file(uploaded, labels)
            if imported.report["accepted_rows"] < 30:
                raise ValueError("at least 30 valid rows are needed for this exploratory split")
            (root / "import-report.json").write_text(json.dumps(imported.report, indent=2) + "\n", encoding="utf-8")
            self._update(job_id, phase="preparing", message="Making a grouped train and holdout split")
            split = make_split(imported.records, FRACTIONS, seed=42)
            summary = {key: imported.report[key] for key in ("accepted_rows", "rejected_rows", "class_counts", "missing_metadata_counts")}
            return self._fit(job_id, root, imported.records, split, labels, model_kind, quality,
                             provenance, license_notice, summary)

        return self._start("train", train)

    def start_preset(self, payload: dict) -> str:
        name = payload.get("preset")
        if not isinstance(name, str) or name not in PRESETS:
            raise ValueError("unknown benchmark preset")
        model_kind, quality = self._settings(payload)

        def train(job_id: str) -> dict:
            root = self.workdir / job_id
            root.mkdir(mode=0o700)
            self._update(job_id, phase="preparing", message="Downloading and verifying pinned public data")
            prepared = prepare_preset(name, self.workdir / "sources")
            summary = {"accepted_rows": len(prepared["records"]), "rejected_rows": 0,
                       "excluded_non_hard_targets": sum(prepared["excluded"].values()),
                       "class_counts": dict(Counter(row["label"] for row in prepared["records"])),
                       "missing_metadata_counts": {}}
            result = self._fit(job_id, root, prepared["records"], prepared["split"], prepared["labels"],
                               model_kind, quality,
                               f"{prepared['name']}; source {prepared['source_config']} at {DATASET_REVISION}; English single-label projection.",
                               prepared["license"], summary, preset=name, ood=prepared["ood"])
            result["benchmark"] = {"name": prepared["name"], "source_revision": DATASET_REVISION,
                                   "source_config": prepared["source_config"], "ood_rows": len(prepared["ood"]),
                                   "excluded_soft_or_unlabeled": prepared["excluded"],
                                   "comparison_note": "Derived fixed-label task. These are not official Open-Jev or JevBench scores."}
            return result
        return self._start("train", train)

    def start_evaluation(self, run_id: str, payload: dict) -> str:
        training = self.run_result(run_id)
        split = payload.get("split")
        if split != "ood" or training.get("preset") is None or set(payload) != {"split"}:
            raise ValueError("this run has no optional shift set")
        def assess(job_id: str) -> dict:
            root = self.workdir / run_id
            self._update(job_id, phase="evaluating", message="Evaluating the current model on the shift set")
            report = evaluate(root / "bundle", root / "ood.jsonl")
            report["source_split"] = "ood"
            report["benchmark_note"] = "Derived fixed-label subset of public synthetic data; not an Open-Jev or JevBench score."
            (root / f"ood-report-{job_id}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report
        return self._start("evaluate", assess, run_id=run_id)

    def predict(self, run_id: str, payload: dict) -> dict:
        self.run_result(run_id)
        if set(payload) != {"text"} or not isinstance(payload["text"], str) or not payload["text"].strip():
            raise ValueError("enter a nonempty text input")
        text = payload["text"]
        if len(text.encode("utf-8")) > 128 * 1024:
            raise ValueError("input text exceeds 128 KiB")
        with self.lock:
            if self.current_model_id != run_id:
                self.current_model = DecisionModel.load(self.workdir / run_id / "bundle", allow_unsigned=True)
                self.current_model_id = run_id
            model = self.current_model
        try:
            return model.predict(text, evaluation=True).to_dict()
        except InvalidInputError as exc:
            raise ValueError(str(exc)) from exc

    def start_benchmark(self, payload: dict, *, run_id: str | None = None) -> str:
        if run_id:
            self.run_result(run_id)
            bundle = self.workdir / run_id / "bundle"
            texts = json.loads((self.workdir / run_id / "benchmark-texts.json").read_text(encoding="utf-8"))
            key = None
            source = "development_partition"
        else:
            bundle_path = payload.get("bundle_path")
            texts = payload.get("texts")
            if not isinstance(bundle_path, str) or not bundle_path.strip():
                raise ValueError("local bundle path is required")
            bundle = Path(bundle_path).expanduser()
            key_path = payload.get("trusted_public_key_path")
            key = Path(key_path).expanduser().read_bytes() if isinstance(key_path, str) and key_path else None
            source = "operator_supplied"
        if (not isinstance(texts, list) or not 1 <= len(texts) <= 1000
                or any(not isinstance(item, str) or not item.strip() or len(item.encode("utf-8")) > 128 * 1024 for item in texts)):
            raise ValueError("representative texts must be 1–1000 nonempty strings")

        def measure(job_id: str) -> dict:
            self._update(job_id, phase="evaluating", message="Loading the bundle and measuring CPU calls")
            result = benchmark(bundle, texts, warmup=20, samples=200, trusted_public_key=key)
            result["workload_sha256"] = sha256(canonical_json(texts))
            result["workload_count"] = len(texts)
            result["workload_source"] = source
            result["sequential_calls_per_second"] = 1000 / statistics.mean(result["raw_timing_ms"])
            result["scope_note"] = "CPU timing on this machine only. Run the same bundle and workload on another machine for a device comparison."
            destination = (self.workdir / run_id) if run_id else self.workdir / "device-benchmarks"
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"timing-{job_id}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return result
        return self._start("benchmark", measure, run_id=run_id)

def make_handler(app: LocalApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            # Do not log request payloads or query strings.
            return

        def _json(self, status: HTTPStatus, value: dict) -> None:
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _allowed(self) -> bool:
            host = self.headers.get("Host")
            origin = self.headers.get("Origin")
            expected = f"127.0.0.1:{self.server.server_port}"
            return host == expected and (origin is None or origin == f"http://{expected}")

        def do_GET(self) -> None:
            if not self._allowed():
                self._json(HTTPStatus.FORBIDDEN, {"error": "invalid local origin"})
                return
            path = urlsplit(self.path).path
            if path == "/api/presets":
                self._json(HTTPStatus.OK, {"revision": DATASET_REVISION,
                                           "presets": {key: {"name": value["name"], "labels": value["labels"],
                                                             "license": value["license"]} for key, value in PRESETS.items()}})
                return
            if path == "/api/runs/current":
                self._json(HTTPStatus.OK, {"run": app.current_run()})
                return
            if path.startswith("/api/jobs/"):
                try:
                    self._json(HTTPStatus.OK, app.job(path.removeprefix("/api/jobs/")))
                except KeyError:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown job"})
                return
            names = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path not in names:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            name, content_type = names[path]
            body = (ASSETS / name).read_bytes()
            if name == "index.html":
                body = body.replace(b"__CSRF_TOKEN__", app.token.encode("ascii"))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; base-uri 'none'; form-action 'self'; object-src 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if not self._allowed() or self.headers.get("X-Decision-Web") != app.token:
                self._json(HTTPStatus.FORBIDDEN, {"error": "invalid local request token"})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON required"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > MAX_UPLOAD_BYTES * 2 + 4096:
                    raise ValueError("request exceeds upload limit")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")
                path = urlsplit(self.path).path
                if path == "/api/preview":
                    result = app.preview(payload)
                    self._json(HTTPStatus.OK, result)
                elif path == "/api/runs":
                    self._json(HTTPStatus.ACCEPTED, {"job_id": app.start_training(payload)})
                elif path.startswith("/api/runs/") and path.endswith("/evaluate"):
                    run_id = path[len("/api/runs/"):-len("/evaluate")]
                    self._json(HTTPStatus.ACCEPTED, {"job_id": app.start_evaluation(run_id, payload)})
                elif path.startswith("/api/runs/") and path.endswith("/predict"):
                    run_id = path[len("/api/runs/"):-len("/predict")]
                    self._json(HTTPStatus.OK, app.predict(run_id, payload))
                elif path.startswith("/api/runs/") and path.endswith("/benchmark"):
                    run_id = path[len("/api/runs/"):-len("/benchmark")]
                    self._json(HTTPStatus.ACCEPTED, {"job_id": app.start_benchmark(payload, run_id=run_id)})
                elif path == "/api/benchmark":
                    self._json(HTTPStatus.ACCEPTED, {"job_id": app.start_benchmark(payload)})
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (ValueError, UnicodeError, json.JSONDecodeError, csv.Error, ImportErrorDetail) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)[:400]})
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown run"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local exploratory decision web app")
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1",
                        help="listen address; use 0.0.0.0 only behind a loopback-published container port")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workdir", type=Path, default=Path("data/web-runs"))
    parser.add_argument("--source", type=Path, default=Path("data/models/all-MiniLM-L6-v2-bc57282"))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    app = LocalApp(args.workdir, args.source)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Decision web app: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
