# Decision Runtime

Apache-2.0 Python SDK for offline, fixed-label text decisions. It loads complete
versioned bundles, verifies hashes and optional Ed25519 signatures, applies the
exported preprocessing, calibration and abstention policy, and runs locally
without a platform connection. It also includes local evaluation, inspection,
benchmark and shadow-mode tools. The optional local lab offers dataset import,
sparse or MiniLM candidate fitting, evaluation and device timing in a browser.

This directory is an independently buildable package. From this runtime directory
(or its separate Git repository):

```bash
python -m pip install -e .
# Add the neural extra to load MiniLM ONNX bundles:
python -m pip install -e '.[neural]'
```

From the platform repository root, use `python -m pip install -e './runtime[neural]'`.

## Local model lab

Install the lab extra and start the app from a local checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[lab]'
.venv/bin/decision-web
```

Open `http://127.0.0.1:8765`. You can upload a UTF-8 CSV/JSONL file with
`text,label` fields, choose a sparse or frozen MiniLM candidate, and evaluate
the exported bundle against a separate labeled file. To tune the last MiniLM
layer, install `.[lab,adapt]` first and choose **Adapted MiniLM**. On a CPU-only
machine, install a compatible CPU PyTorch wheel before the `adapt` extra.
MiniLM source files are downloaded at a pinned revision and verified by hash.

The app also offers two pinned public Open-Jev **derived fixed-label** presets:
English email kind (six labels) and English support routing (four labels). The
first use downloads and verifies public synthetic files from Hugging Face; no
benchmark rows or model weights are included in this repository. You can run
the publisher's test and OOD subsets after fitting. These are software and
synthetic-data diagnostics, **not official Open-Jev or JevBench scores** or
evidence of real customer accuracy. See [BENCHMARKS.md](BENCHMARKS.md) for
source rights, selection rules, split counts and limitations.

The separate **JevBench public-task scorecard** accepts typed-model predictions
as JSONL. Each row needs `task_id` and either `probabilities` keyed by the
task's exact option labels or a `label` for a label-only system. It also accepts
the official runner's `probs_as_returned` field. Download the pinned public
tasks from JevBench, run a compatible typed model, and upload its predictions
to the scorecard. The app rechecks validity and scores all 231 public tasks,
counting missing or malformed answers as wrong. Example input row:

```json
{"task_id":"original-policy-01-0","probabilities":{"no":0.9,"yes":0.1},"latency_ms":12.3}
```

The scorecard has per-tier and per-type accuracy, strict vector validity,
calibration diagnostics and optional self-reported latency. It cannot run the
fixed-label candidate above on JevBench, access sealed tasks, or issue the
official JevBench composite score. See [BENCHMARKS.md](BENCHMARKS.md).

The **Device timing** section measures cold model load, 200 sequential CPU
predictions after 20 warmups, p50/p95 latency, throughput and peak process RSS.
For another machine, copy the *same* bundle and the run's
`benchmark-texts.json`, start this app there, and use **Time a copied bundle**.
Download both JSON reports and select them in **Compare timing reports**. The
app compares reports only when bundle manifest hash, workload hash and sample
count match. Measurements depend on the machine, software environment and
workload. They do not claim phone, GPU or hosted-device performance.

The server binds only to `127.0.0.1` and saves uploads, candidate bundles and
reports in `data/web-runs/` under the current directory by default. Run it on
a trusted machine with permitted data; it has one local operator, no account
isolation and no hosted upload service. Training and evaluation are exploratory.
They do not issue a qualified production release.

For a signed bundle, obtain its trusted public key separately from the bundle:

```python
from decision_runtime import DecisionModel

model = DecisionModel.load("router-v1", trusted_public_key=open("trusted-public.pem", "rb").read())
result = model.predict("Please help with my invoice")
print(result.to_dict())
```

For an unsigned bundle created locally, pass `allow_unsigned=True` explicitly.
Only a qualified release can return an actionable `accepted` choice. Evaluation
mode returns `would_accept` or `would_defer` without an actionable choice.

```bash
decision-inspect router-v1 --trusted-public-key trusted-public.pem
decision-evaluate router-v1 labeled.csv --trusted-public-key trusted-public.pem --output evaluation.json
decision-benchmark router-v1 --texts representative-texts.json --output benchmark.json
```

`labeled.csv` needs `text,label` columns. JSONL accepts objects with the same
fields. The evaluator uses the local inference kernel and reports label accuracy,
macro F1, coverage, accepted error, calibration and a confusion matrix. It does
not claim independent sampling or production qualification. The format contract
is in [BUNDLE_FORMAT.md](BUNDLE_FORMAT.md). A runnable sparse-bundle example is
in [examples](examples).

The package contains no telemetry or model weights. Customer bundles and data
remain under their own permissions and notices.

Run the standalone contract tests with `python -m unittest discover -s tests`
from this directory. A wheel can be built with
`python -m pip wheel --no-deps . -w dist`.
