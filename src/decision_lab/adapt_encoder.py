"""One bounded supervised MiniLM adaptation: tune its last transformer layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from .model_source import ADAPTATION_WEIGHTS_SHA256, MODEL_ID, REVISION, SOURCE_FILES, verify_source
from .splits import load_records, validate_split


def adapt_encoder(dataset: Path, split_path: Path, source: Path, output: Path,
                  *, max_train_rows: int = 512, epochs: int = 1, batch_size: int = 8) -> dict:
    """Export adapted encoder weights to ONNX; classifier fitting remains separate."""
    if output.exists():
        raise ValueError("adapted source output already exists")
    if not 20 <= max_train_rows <= 4096 or not 1 <= epochs <= 3 or not 1 <= batch_size <= 32:
        raise ValueError("adaptation exceeds the supported bounded recipe")
    verify_source(source)
    weights = source / "model.safetensors"
    if weights.is_symlink() or not weights.is_file():
        raise ValueError("pinned safetensors weights are missing")
    with weights.open("rb") as weight_file:
        if hashlib.file_digest(weight_file, "sha256").hexdigest() != ADAPTATION_WEIGHTS_SHA256:
            raise ValueError("pinned safetensors hash mismatch")

    import numpy as np
    import onnxruntime as ort
    import torch
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoModel

    rows = load_records(dataset)
    split_raw = split_path.read_bytes()
    split = json.loads(split_raw)
    partitions = validate_split(rows, split)
    labels = sorted({row["label"] for row in partitions["train"]})
    if not 2 <= len(labels) <= 20:
        raise ValueError("training partition needs 2–20 labels")
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    tokenizer.no_truncation()
    by_label: dict[str, list[tuple[dict, object]]] = defaultdict(list)
    for row in partitions["train"]:
        encoded = tokenizer.encode(row["text"], add_special_tokens=True)
        if 2 < len(encoded.ids) <= 256 and len(row["text"].encode("utf-8")) <= 1024 * 1024:
            by_label[row["label"]].append((row, encoded))
    if any(not by_label[label] for label in labels):
        raise ValueError("some labels have no supported adaptation examples")
    rng = random.Random(0)
    for items in by_label.values():
        rng.shuffle(items)
    selected = []
    while len(selected) < max_train_rows and any(by_label.values()):
        for label in labels:
            if by_label[label] and len(selected) < max_train_rows:
                selected.append(by_label[label].pop())
    rng.shuffle(selected)
    if len(selected) < 20:
        raise ValueError("too few supported adaptation examples")

    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = AutoModel.from_pretrained(str(source), local_files_only=True, use_safetensors=True,
                                      trust_remote_code=False, attn_implementation="eager")
    for parameter in model.parameters():
        parameter.requires_grad = False
    last_layer = model.encoder.layer[-1]
    for parameter in last_layer.parameters():
        parameter.requires_grad = True
    head = torch.nn.Linear(384, len(labels))
    optimizer = torch.optim.AdamW([
        {"params": list(last_layer.parameters()), "lr": 3e-5},
        {"params": list(head.parameters()), "lr": 1e-3},
    ], weight_decay=0.01)
    model.train()
    head.train()
    losses = []
    for _ in range(epochs):
        rng.shuffle(selected)
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            length = max(len(encoded.ids) for _, encoded in batch)
            ids = np.zeros((len(batch), length), dtype=np.int64)
            masks = np.zeros_like(ids)
            types = np.zeros_like(ids)
            for index, (_, encoded) in enumerate(batch):
                size = len(encoded.ids)
                ids[index, :size] = encoded.ids
                masks[index, :size] = encoded.attention_mask
                types[index, :size] = encoded.type_ids
            input_ids = torch.from_numpy(ids)
            attention_mask = torch.from_numpy(masks)
            token_type_ids = torch.from_numpy(types)
            targets = torch.tensor([labels.index(row["label"]) for row, _ in batch], dtype=torch.long)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
            pooled = (outputs * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            loss = torch.nn.functional.cross_entropy(head(pooled), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

    model.eval()

    class ExportEncoder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        working = Path(temporary) / "source"
        (working / "onnx").mkdir(parents=True)
        shutil.copyfile(source / "tokenizer.json", working / "tokenizer.json")
        sample = torch.tensor([[101, 2023, 2003, 1037, 7953, 102] + [0] * 10] * 2, dtype=torch.long)
        mask = torch.ones_like(sample)
        types = torch.zeros_like(sample)
        batch_dimension = torch.export.Dim("batch", min=1, max=32)
        sequence_dimension = torch.export.Dim("sequence", min=2, max=256)
        dynamic_shapes = {name: {0: batch_dimension, 1: sequence_dimension}
                          for name in ("input_ids", "attention_mask", "token_type_ids")}
        with torch.no_grad():
            torch.onnx.export(
                ExportEncoder(model).eval(), (sample, mask, types), str(working / "onnx/model.onnx"),
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["last_hidden_state"], opset_version=18, dynamo=True,
                dynamic_shapes=dynamic_shapes, external_data=False,
            )
        # Check the exported graph at several lengths before making it available
        # for head fitting. Quality evaluation itself also uses this ONNX graph.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        exported = ort.InferenceSession(str(working / "onnx/model.onnx"), sess_options=options,
                                        providers=["CPUExecutionProvider"])
        ordered = sorted((encoded for _, encoded in selected), key=lambda item: len(item.ids))
        parity_inputs = [ordered[0], ordered[len(ordered) // 2], ordered[-1],
                         tokenizer.encode("word " * 254, add_special_tokens=True)]
        max_conversion_abs_diff = 0.0
        parity_differences = []
        with torch.no_grad():
            for encoded in parity_inputs:
                ids = torch.tensor([encoded.ids], dtype=torch.long)
                masks = torch.tensor([encoded.attention_mask], dtype=torch.long)
                types = torch.tensor([encoded.type_ids], dtype=torch.long)
                reference = ExportEncoder(model)(ids, masks, types).detach().numpy()
                actual = exported.run(None, {"input_ids": ids.numpy(), "attention_mask": masks.numpy(),
                                             "token_type_ids": types.numpy()})[0]
                if actual.shape != reference.shape:
                    raise ValueError("adapted ONNX export changed encoder output shape")
                difference = float(np.max(np.abs(actual - reference)))
                parity_differences.append({"tokens": len(encoded.ids), "max_abs_diff": difference})
                max_conversion_abs_diff = max(max_conversion_abs_diff, difference)
        if not np.isfinite(max_conversion_abs_diff) or max_conversion_abs_diff > 1e-3:
            raise ValueError(f"adapted ONNX export exceeds conversion parity tolerance: {parity_differences}")
        source_hashes = {}
        for name in ("onnx/model.onnx", "tokenizer.json"):
            with (working / name).open("rb") as item:
                source_hashes[name] = hashlib.file_digest(item, "sha256").hexdigest()
        info = {
            "recipe": "last_layer_supervised_v1", "base_model_id": MODEL_ID,
            "base_model_revision": REVISION, "base_model_weights_sha256": ADAPTATION_WEIGHTS_SHA256,
            "base_source_file_hashes": SOURCE_FILES, "source_file_hashes": source_hashes,
            "dataset_sha256": split["dataset_sha256"],
            "split_manifest_sha256": hashlib.sha256(split_raw).hexdigest(),
            "training_rows": len(selected), "max_train_rows": max_train_rows,
            "epochs": epochs, "batch_size": batch_size, "seed": 0,
            "mean_training_loss": sum(losses) / len(losses),
            "max_conversion_abs_diff": max_conversion_abs_diff,
            "conversion_parity_tolerance": 1e-3,
            "conversion_parity_samples": parity_differences,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        }
        (working / "adaptation.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise ValueError("adapted source path appeared during training")
        working.rename(output)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt the pinned MiniLM encoder with one supervised last-layer recipe")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    try:
        info = adapt_encoder(args.dataset, args.split, args.source, args.output,
                             max_train_rows=args.max_train_rows, epochs=args.epochs, batch_size=args.batch_size)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"adaptation failed: {exc}\n")
    print(json.dumps({"output": str(args.output), "training_rows": info["training_rows"],
                      "mean_training_loss": info["mean_training_loss"]}))


if __name__ == "__main__":
    main()
