"""Milestone 1: the feedback categories, the hand-written dataset, the split, and the training template.

All comments in `dataset/raw/<category>.txt` were invented for this project (one comment per line).
`build_dataset` validates them, writes `dataset/feedback.csv` and a stratified train/test split.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "raw"
OUT_DIR = ROOT / "dataset"

# slug -> what the category means (the slug is also the name inside the special token: <slug>)
CATEGORIES: dict[str, str] = {
    "product_issue": "Defective, damaged, wrong, spoiled or poorly made goods",
    "staff_praise": "Positive remarks about the behaviour or helpfulness of staff",
    "staff_complaint": "Rude, unhelpful, dishonest or untrained staff",
    "pricing_value": "Prices, fees, discounts, billing and value for money",
    "store_experience": "The place itself: cleanliness, layout, crowds, queues, facilities, parking",
}
LABELS: list[str] = list(CATEGORIES)

PROMPT_PREFIX = "Comment: "
SEPARATOR = "\nCategory:"  # the label token comes straight after this, then EOS
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Example:
    text: str
    label: str


def label_token(label: str) -> str:
    return f"<{label}>"


# ---------------------------------------------------------------- loading / splitting

def load_raw(raw_dir: Path = RAW_DIR) -> list[Example]:
    """Read one file per category; drop blanks and case-insensitive duplicates."""
    examples: list[Example] = []
    seen: set[str] = set()
    for label in LABELS:
        path = Path(raw_dir) / f"{label}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            text = " ".join(line.split())
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            examples.append(Example(text, label))
    return examples


def stratified_split(examples: list[Example], test_frac: float = 0.2,
                     seed: int = 42) -> tuple[list[Example], list[Example]]:
    """Split so every category keeps (about) the same share in train and test."""
    by_label: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_label[ex.label].append(ex)
    rng = random.Random(seed)
    train: list[Example] = []
    test: list[Example] = []
    for label in LABELS:
        items = by_label.get(label, [])[:]
        rng.shuffle(items)
        n_test = max(1, round(len(items) * test_frac)) if items else 0
        test += items[:n_test]
        train += items[n_test:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def class_balance(examples: list[Example]) -> dict[str, int]:
    counts = Counter(ex.label for ex in examples)
    return {label: counts.get(label, 0) for label in LABELS}


def build_dataset(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR,
                  test_frac: float = 0.2, seed: int = 42) -> dict:
    """Write feedback.csv, train.jsonl, test.jsonl and stats.json; return the stats."""
    examples = load_raw(raw_dir)
    train, test = stratified_split(examples, test_frac, seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "feedback.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["comment", "label"])
        writer.writerows((ex.text, ex.label) for ex in examples)
    for name, rows in (("train", train), ("test", test)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps({"text": ex.text, "label": ex.label}, ensure_ascii=False) + "\n")
    stats = {"total": len(examples), "train": len(train), "test": len(test), "seed": seed,
             "class_balance": class_balance(examples),
             "train_balance": class_balance(train), "test_balance": class_balance(test)}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_split(name: str, out_dir: Path = OUT_DIR) -> list[Example]:
    path = Path(out_dir) / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m app.cli dataset")
    return [Example(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------- the template (training AND inference)

def encode_prompt(tokenizer, text: str, max_length: int = 128) -> list[int]:
    """`Comment: <text>\\nCategory:` as ids. The comment is truncated so the separator always survives.

    Training and inference both call this, so the model sees the identical template.
    """
    sep_ids = tokenizer(SEPARATOR, add_special_tokens=False)["input_ids"]
    head = tokenizer(PROMPT_PREFIX + " ".join(text.split()), add_special_tokens=False)["input_ids"]
    head = head[: max(1, max_length - len(sep_ids) - 2)]
    return head + sep_ids


def encode_example(tokenizer, text: str, label_token_id: int, max_length: int = 128) -> dict:
    """prompt + <label> + EOS. Loss is computed on the label token only; every other position is -100."""
    prompt = encode_prompt(tokenizer, text, max_length)
    input_ids = prompt + [label_token_id, tokenizer.eos_token_id]
    labels = [IGNORE_INDEX] * len(prompt) + [label_token_id, IGNORE_INDEX]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class FeedbackDataset:
    """A torch-style dataset (len + getitem) of templated, masked examples."""

    def __init__(self, examples: list[Example], tokenizer, max_length: int = 128):
        self.items = [encode_example(tokenizer, ex.text, tokenizer.convert_tokens_to_ids(label_token(ex.label)),
                                     max_length) for ex in examples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class PadCollator:
    """Right-pad a batch; padded labels are -100 so they never contribute to the loss."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        width = max(len(b["input_ids"]) for b in batch)

        def pad(key: str, value: int):
            return torch.tensor([b[key] + [value] * (width - len(b[key])) for b in batch], dtype=torch.long)

        return {"input_ids": pad("input_ids", self.pad_id), "attention_mask": pad("attention_mask", 0),
                "labels": pad("labels", IGNORE_INDEX)}
