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

## JevBench public-task scorecard

The web app can also score outputs from a **separate typed-decision model** on
the 231 public JevBench tasks. It downloads and hash-checks the original 72,
easy 48 and hard 111 files from JevBench commit
`1bcc55eb6c8cffde2306b3db03ede39b61c6152a`. The
[JevBench repository](https://github.com/fstandhartinger/jevbench) and its
[hard-tier method](https://github.com/fstandhartinger/jevbench/blob/main/datasets/HARD-TIER.md)
mark these public tasks MIT; no task content is bundled here.

Upload JSONL with one row per task and exact option probabilities. The scorer
follows JevBench's public per-item rules: exact label keys, finite values in
`[0,1]`, strict sum tolerance `0.001`, normalization only for rounding within
`0.02`, and lexicographic tie breaking. Invalid vectors and missing tasks
count as incorrect over the full public denominator. Label-only answers can
receive accuracy but never invented calibration. The app recomputes scores
from predictions and ignores submitted `correct` fields. It reports public
tier/type accuracy, validity, Brier score and 10-bin ECE; any uploaded timing
is explicitly self-reported. Predictions and reports stay in the local ignored
work directory.

The current official JevBench ranking uses sealed tasks, calibration, speed
and cost. A public-subset scorecard cannot reproduce that ranking, establish
model generalization or measure cost without provider billing. The local
fixed-label model does not accept per-item rubrics, so it cannot use this
typed scorecard directly. Use a typed-model adapter and the upstream harness
for an official JevBench submission.
