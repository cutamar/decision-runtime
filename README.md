# Decision Runtime

Apache-2.0 Python SDK for offline, fixed-label text decisions. It loads complete
versioned bundles, verifies hashes and optional Ed25519 signatures, applies the
exported preprocessing, calibration and abstention policy, and runs locally
without a platform connection. It also includes local evaluation, inspection,
benchmark and shadow-mode tools.

This directory is an independently buildable package. From the parent project:

```bash
python -m pip install -e ./runtime
# Add the neural extra to load MiniLM ONNX bundles:
python -m pip install -e './runtime[neural]'
```

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

Run the standalone contract tests with `python -m unittest discover -s runtime/tests`
from the parent directory. A wheel can be built with
`python -m pip wheel --no-deps ./runtime -w runtime/dist`.
