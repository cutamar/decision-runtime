# Decision bundle format 1.0

The machine-readable [schemas](src/decision_runtime/schemas) cover the manifest, prediction and
labeled evaluation row. The runtime loader remains the authoritative validator
for cross-file hashes and model-specific constraints.

The bundle is a directory of regular files. The loader rejects links, missing
files, extra or missing manifest entries, oversized files, unsupported formats,
and changed hashes. JSON files are UTF-8. `manifest.json` has `format_version`
`"1.0"`, an `artifact_kind`, nonempty `model_version` and `policy_version`, a
`trust` mode, `files` mapping each required relative path to its lowercase
SHA-256 hex digest, and `inference_payload_sha256`.

| Kind | Inference payload files |
| --- | --- |
| `sparse_linear` | `linear.json`, `vectorizer.json`, `labels.json`, `preprocessing.json`, `calibration.json`, `policy.json` |
| `encoder_onnx` | `model.onnx`, `tokenizer.json`, `head.json`, `labels.json`, `preprocessing.json`, `calibration.json`, `policy.json` |

Both kinds also require `qualification.json`, `evaluation.json`, `lineage.json`,
`model-card.md`, and `licenses/NOTICE.txt`. An encoder bundle also requires
`licenses/Apache-2.0.txt` for its pinned pretrained model. The fixed `files`
mapping must contain exactly the required files for that kind.

The payload digest is SHA-256 of canonical JSON mapping inference payload
filenames to their file hashes. Canonical JSON uses sorted keys, compact
separators, UTF-8, and no NaN values. Manifest files are canonical JSON with
one trailing newline. A signed bundle has `trust: "signed_ed25519"` and a
64-byte raw `manifest.sig` containing an Ed25519 signature over the canonical
manifest bytes *without* the newline. The verification key is supplied by the
caller from outside the bundle. A local unsigned bundle has `trust:
"local_unsigned"`, no signature, and requires an explicit `allow_unsigned=True`.

`qualification.json` binds `model_version`, the inference payload digest, the
SHA-256 of `evaluation.json`, and its status to the manifest. A loader rejects
any mismatch. `qualified` is the only status that permits normal acceptance;
other statuses remain non-actionable. A signature proves origin and integrity,
not quality.

`labels.json` has an ordered `ids` array of 2–20 unique label IDs. Output
probabilities use this order. `calibration.json` carries a positive temperature
and a `fitted` or `unfitted` status. `policy.json` carries one threshold per
label, `review_only` IDs, a tie tolerance, and a fallback identifier. All
preprocessing and fitted parameters needed to reproduce predictions are in the
payload files. Sparse bundles contain a TF-IDF vocabulary and linear scores;
encoder bundles contain the ONNX graph, tokenizer, linear head, and pooling
configuration. For exact accepted values, see the version 1.0 loader; changes
to this contract require a new format version.

Prediction schema 1.0 contains `status`, `choice`, `suggested_choice`,
`top_probability`, `probabilities`, `abstention_reason`, `model_version`,
`policy_version`, and `calibration_status`. Normal mode returns `accepted` or
`abstain`; evaluation mode returns `would_accept` or `would_defer` and never an
actionable `choice`. Empty or invalid request text raises `InvalidInputError`.
Supported but overlong input abstains with `input_too_long`.

Compatibility target: Python 3.11+ on Linux x86-64 CPU. The encoder path uses
the optional neural dependencies. Bundle versions are immutable; customers
should pin a bundle and an independently distributed public key.
