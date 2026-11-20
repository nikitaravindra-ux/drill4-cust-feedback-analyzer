"""Milestone 4: classify a comment by SCORING the category tokens, not by free generation.

One forward pass over the same template used in training (`Comment: ...\\nCategory:`), take the logits at
the last position, keep only the category-token ids, softmax them. Deterministic, fast, and the answer
can never be anything but one of the defined categories.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.dataset import LABELS, encode_prompt
from app.device import pick_device
from app.finetune import MODEL_DIR
from app.tokens import category_token_ids


class ModelNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class Prediction:
    label: str
    scores: dict[str, float]  # softmax over the category tokens only; sums to 1

    @property
    def confidence(self) -> float:
        return self.scores[self.label]


class Classifier:
    def __init__(self, model, tokenizer, device: str = "cpu", max_length: int = 128):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.label_ids = category_token_ids(tokenizer)

    def classify(self, text: str) -> Prediction:
        import torch
        if not text or not text.strip():
            raise ValueError("Empty comment")
        ids = encode_prompt(self.tokenizer, text, self.max_length)
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([ids], device=self.device)).logits[0, -1]
        probs = torch.softmax(logits[self.label_ids].float(), dim=-1).cpu().tolist()
        scores = dict(zip(LABELS, probs))
        return Prediction(label=max(scores, key=scores.get), scores=scores)


def load_classifier(model_dir: str | Path = MODEL_DIR, device: str | None = None) -> Classifier:
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise ModelNotFoundError(
            f"No trained model in {model_dir}. Train one first:  python -m app.cli train")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    return Classifier(model, tokenizer, pick_device(device))
