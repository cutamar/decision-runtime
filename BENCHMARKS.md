# Public benchmark presets

The [Open-Jev benchmark index](https://zefan-cai.github.io/open-jev/benchmarks/)
is a collection of different evaluations. Its JevBench section reports
Open-Jev's run on [JevBench](https://github.com/fstandhartinger/jevbench), an
independent typed-decision benchmark. The historical Open-Jev comparison uses
JevBench's 231 public tasks out of 534 total. The current JevBench ranking has
since evolved and includes sealed decisions plus intelligence, calibration,
speed and cost axes. Its tasks include Noul, Choice and Score decisions using
a state, rubric and per-item options. This fixed-label text classifier cannot
produce an official JevBench result. The index's Mailroom 921 control is a separate
87-email, 11-head provider probe; our email preset is not that probe.

The local app uses two narrow projections of the
[Open-Jev dataset](https://huggingface.co/datasets/ZefanCai/Open-Jev) at revision
`c67699e13d0ae25e35b77165a4b6b079bedc8aba`. Each downloaded gzip file
is SHA-256 checked against a pin in `decision_lab/open_jev.py`. No original
question descriptions, rubrics, audit metadata or target probabilities enter
model text. We keep only English, hard single-label targets whose complete
option set maps to a fixed list of label IDs. Soft targets are excluded and
counted. We preserve publisher train, calibration, test and OOD partitions;
publisher validation families are divided deterministically between our
development and policy partitions. No family crosses partitions.

| App preset | Publisher source and head | Labels | Train / dev / calibration / policy / test / OOD rows |
| --- | --- | --- | --- |
| Email kind | `mailroom-control-v1`, `kind` | account_statement, invoice, newsletter, other, payment_confirmation, promotion | 2,304 / 90 / 126 / 99 / 261 / 720 |
| Support routing | `release-v2-redistributable` customer-control-v1, `category` | account, billing, bug_report, feature_request | 545 / 12 / 37 / 13 / 75 / 79 |

The [mailroom source card](https://huggingface.co/datasets/ZefanCai/Open-Jev/blob/main/cards/mailroom-control-v1.md)
marks the original generated email and labels CC0-1.0. The main dataset card
marks original generated customer-control records CC0-1.0, while noting
unverified rights for some upstream short question descriptions. The support
projection copies only the state text and hard label IDs, excluding those
descriptions. The adapter's source and label mapping are in this repository;
the downloaded records remain in the user's local work directory.

The evaluator reports supported accuracy, macro F1, per-label precision and
recall, confusion, negative log likelihood, multiclass Brier score, 10-bin
expected calibration error, would-accept coverage and accepted error. Its
published test/OOD outputs record the exact bundle manifest and labeled-file
hashes. These are appropriate diagnostics for a fixed-label System One
classifier; they are not interchangeable with JevBench's typed-response and
sealed-item protocol.

These sets are synthetic and publicly labeled. Related templates, correlated
families and visible public answers limit independent evidence, even with
group-aware partitions. Inspect per-class results and uncertainty; a high
score is not evidence that a model works on company email or tickets. For a
customer pilot, use authorized real examples, frozen requirements, a current
routing baseline and an untouched confirmation set. Device timing measures
only the exported bundle on the machine running the app.
