"""Specialized-vs-frontier benchmark on the public Open-Jev mailroom task.

Trains tiny per-head classifiers (a frozen multilingual MiniLM encoder plus one
linear head per decision head) on the public Open-Jev ``mailroom-control-v1``
data, then evaluates on the held-out ``test`` split with the same hard one-hot
scoring the Open-Jev benchmark uses. It reports per-head, per-language, and
overall accuracy next to the published frontier numbers.

Honest scope, please read before quoting any number:
  * This is a DIFFERENT item sample from Open-Jev's frozen 87-request / 921
    decision provider probe, which is not public and not reconstructable from
    the released config. Treat this as "same task", not "identical items".
  * The data is synthetic and templated, so it is easier than real inbox mail.
    The script asserts there is no email/family/text overlap between train and
    test, so the score is leakage-free, but real-world accuracy will be lower.
  * Our model is task-specialized; Jev and GPT are general models. That trade
    (train per task, fixed labels) is the whole point.

Published reference (frozen 921-decision probe, from the Open-Jev docs):
    Jev 1.13.0   908/921 = 98.6%   (en 347/351, zh 279/285, tr 282/285)
    GPT-6 Astra  913/921 = 99.1%
    GPT-5.6 Luna 900/921 = 97.7%

Requirements:
    pip install -e '.[lab,adapt]' sentencepiece
    (torch, transformers, scikit-learn, numpy, plus the multilingual model
    downloaded from Hugging Face on first run.)

Run:
    python examples/mailroom_multilingual_benchmark.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

import numpy as np

DATASET_REPO = "ZefanCai/Open-Jev"
DATASET_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
CONFIG = "mailroom-control-v1"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE = Path("data/open-jev-mailroom")
# SHA-256 of each raw split file at the pinned dataset revision.
HASHES = {
    "train": "ebbfae0b4024ac8dbbbc709f54b8cc02023d5953a0b712888a9d2ffbb5189811",
    "test": "d6662f7f7870d3bc629956f07fed8dfb5f0a42d1a923fbbe764db30e933036be",
}
JEV = {"en": (347, 351), "zh": (279, 285), "tr": (282, 285), "overall": (908, 921)}


def fetch(split: str) -> Path:
    dst = CACHE / f"{split}.jsonl.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and hashlib.sha256(dst.read_bytes()).hexdigest() == HASHES[split]:
        return dst
    url = (f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}"
           f"/raw/{CONFIG}/{split}.jsonl.gz")
    data = urlopen(url, timeout=180).read()
    if hashlib.sha256(data).hexdigest() != HASHES[split]:
        raise ValueError(f"hash mismatch for {split}")
    dst.write_bytes(data)
    return dst


def render_email(state: dict) -> str | None:
    email = state.get("email") if isinstance(state, dict) else None
    if not isinstance(email, dict):
        return None
    frm = email.get("from") if isinstance(email.get("from"), dict) else {}
    subject, body, date = email.get("subject"), email.get("body"), email.get("date")
    if not (isinstance(subject, str) and subject and isinstance(body, str) and body):
        return None
    return f"Subject: {subject}\nFrom: {frm.get('display_name')} <{frm.get('email')}>\nDate: {date}\n\n{body}"


def load(split: str):
    emails: dict[str, tuple[str, str, str]] = {}   # email id -> (text, language, group)
    labels: dict[str, dict[str, str]] = defaultdict(dict)  # head -> {email id: label}
    with gzip.open(fetch(split), "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            row_id = str(record.get("id", ""))
            if ":" not in row_id:
                continue
            email_id, head = row_id.rsplit(":", 1)
            target, options = record.get("target"), record.get("options")
            if not (isinstance(target, list) and isinstance(options, list) and target.count(1.0) == 1
                    and all(value in (0, 0.0, 1, 1.0) for value in target)):
                continue  # skip soft or inapplicable (no-gold) decisions
            option_labels = [option.partition(":")[0] for option in options if isinstance(option, str)]
            if len(option_labels) != len(options):
                continue
            if email_id not in emails:
                text = render_email(record.get("state"))
                if not text:
                    continue
                metadata = record.get("metadata") or {}
                emails[email_id] = (text, metadata.get("language"), record.get("group_id"))
            labels[head][email_id] = option_labels[target.index(1.0)]
    return emails, labels


def main() -> None:
    import torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModel, AutoTokenizer

    print("Loading public mailroom splits...", flush=True)
    train_emails, train_labels = load("train")
    test_emails, test_labels = load("test")

    # Leakage guard: the held-out test must share no email, family, or exact text.
    train_ids, test_ids = set(train_emails), set(test_emails)
    train_groups = {g for _, _, g in train_emails.values() if g}
    test_groups = {g for _, _, g in test_emails.values() if g}
    train_text = {t for t, _, _ in train_emails.values()}
    assert not (train_ids & test_ids), "email id overlap between train and test"
    assert not (train_groups & test_groups), "family/group overlap between train and test"
    assert not (train_text & {t for t, _, _ in test_emails.values()}), "exact text overlap"
    print(f"train emails={len(train_emails)}  test emails={len(test_emails)}  "
          f"test families={len(test_groups)}  heads={len(set(train_labels) & set(test_labels))}", flush=True)
    print("Leakage check passed: no email / family / text overlap.\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    encoder = AutoModel.from_pretrained(MODEL).eval()
    torch.set_num_threads(4)

    def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=256, return_tensors="pt")
            with torch.no_grad():
                hidden = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), p=2, dim=1)
            vectors.append(pooled.numpy())
        return np.vstack(vectors)

    started = time.time()
    train_ids = list(train_emails)
    test_ids = list(test_emails)
    print("Embedding emails with the frozen multilingual MiniLM...", flush=True)
    train_vectors = embed([train_emails[i][0] for i in train_ids])
    test_vectors = embed([test_emails[i][0] for i in test_ids])
    train_index = {email_id: i for i, email_id in enumerate(train_ids)}
    test_index = {email_id: i for i, email_id in enumerate(test_ids)}
    print(f"Embedded {len(train_ids) + len(test_ids)} emails in {time.time() - started:.0f}s.\n", flush=True)

    heads = sorted(set(train_labels) & set(test_labels))
    per_head, lang_correct, lang_total = {}, Counter(), Counter()
    overall_correct = overall_total = 0
    for head in heads:
        train_head, test_head = train_labels[head], test_labels[head]
        classifier = LogisticRegression(max_iter=1000).fit(
            np.array([train_vectors[train_index[e]] for e in train_head]),
            [train_head[e] for e in train_head])
        ids = list(test_head)
        predictions = classifier.predict(np.array([test_vectors[test_index[e]] for e in ids]))
        gold = [test_head[e] for e in ids]
        correct = int(sum(p == y for p, y in zip(predictions, gold)))
        per_head[head] = (correct, len(gold))
        overall_correct += correct
        overall_total += len(gold)
        for email_id, prediction, y in zip(ids, predictions, gold):
            language = test_emails[email_id][1]
            lang_total[language] += 1
            lang_correct[language] += int(prediction == y)

    print("Per head (held-out public test):")
    for head in heads:
        correct, total = per_head[head]
        print(f"  {head:26s} {correct}/{total} = {100 * correct / total:.1f}%")
    print("\nPer language        ours (public test)      Jev 1.13.0 (frozen probe)")
    for language in ("en", "zh", "tr"):
        jev_correct, jev_total = JEV[language]
        ours = f"{lang_correct[language]}/{lang_total[language]} = {100 * lang_correct[language] / lang_total[language]:.2f}%"
        print(f"  {language:6s} {ours:>28s} {f'{jev_correct}/{jev_total} = {100 * jev_correct / jev_total:.1f}%':>24s}")
    jev_correct, jev_total = JEV["overall"]
    ours = f"{overall_correct}/{overall_total} = {100 * overall_correct / overall_total:.2f}%"
    print(f"  {'ALL':6s} {ours:>28s} {f'{jev_correct}/{jev_total} = {100 * jev_correct / jev_total:.1f}%':>24s}")


if __name__ == "__main__":
    main()
