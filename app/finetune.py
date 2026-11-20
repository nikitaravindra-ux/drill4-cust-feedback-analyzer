"""Milestone 3: fine-tune GPT-2 with the Hugging Face Trainer so it emits the right category token."""
from __future__ import annotations

import json
from pathlib import Path

from app.dataset import ROOT, FeedbackDataset, PadCollator, load_split
from app.device import pick_device
from app.tokens import add_category_tokens

MODEL_DIR = ROOT / "models" / "feedback-gpt2"
DEFAULT_BASE = "distilgpt2"


def training_arguments(output_dir: str, device: str = "cpu", epochs: float = 8, batch_size: int = 8,
                       lr: float = 1e-4, max_steps: int = -1, seed: int = 42):
    """The TrainingArguments used for the real run (and, with max_steps, for the toy test run)."""
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=0.1,           # a float < 1 means a fraction of all steps
        lr_scheduler_type="linear",
        logging_steps=5,
        save_strategy="no",          # we save the final model ourselves
        eval_strategy="no",          # the test split is only touched by `evaluate`
        report_to="none",
        seed=seed,
        use_cpu=(device == "cpu"),   # otherwise Trainer picks mps/cuda by itself
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )


def build_trainer(model, tokenizer, examples, args):
    from transformers import Trainer
    dataset = FeedbackDataset(examples, tokenizer)
    return Trainer(model=model, args=args, train_dataset=dataset,
                   data_collator=PadCollator(tokenizer.pad_token_id))


def mean_loss(trainer, tokenizer) -> float:
    """Average label-token loss over the trainer's training set (no gradient)."""
    import torch
    model, ds = trainer.model, trainer.train_dataset
    collate = trainer.data_collator
    device = next(model.parameters()).device
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ds), 8):
            batch = {k: v.to(device) for k, v in collate([ds[j] for j in range(i, min(i + 8, len(ds)))]).items()}
            total += model(**batch).loss.item() * len(batch["input_ids"])
            n += len(batch["input_ids"])
    return total / n


def train(base: str = DEFAULT_BASE, output_dir: Path = MODEL_DIR, epochs: float = 8, batch_size: int = 8,
          lr: float = 1e-4, device: str | None = None, seed: int = 42) -> dict:
    """Download `base`, add the category tokens, fine-tune on dataset/train.jsonl, save to `output_dir`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(seed)
    device = pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token
    model = AutoModelForCausalLM.from_pretrained(base)
    add_category_tokens(tokenizer, model)      # milestone 2: tokens + resize + mean init
    examples = load_split("train")
    args = training_arguments(str(output_dir), device, epochs, batch_size, lr, seed=seed)
    trainer = build_trainer(model, tokenizer, examples, args)
    result = trainer.train()
    history = [h for h in trainer.state.log_history if "loss" in h]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    info = {"base": base, "device": device, "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "train_examples": len(examples), "final_train_loss": result.training_loss,
            "loss_history": [{"step": h["step"], "loss": h["loss"]} for h in history]}
    (output_dir / "training_info.json").write_text(json.dumps(info, indent=2))
    return info
