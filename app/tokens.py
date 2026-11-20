"""Milestone 2: add one special token per category to the tokenizer AND resize the model's embeddings."""
from __future__ import annotations

from app.dataset import LABELS, label_token

LABEL_TOKENS: list[str] = [label_token(label) for label in LABELS]


def add_category_tokens(tokenizer, model, init_noise: float = 1e-3) -> list[int]:
    """Register the category tokens and make the model able to embed them.

    1. tokenizer.add_special_tokens(...)           - new tokens get new ids at the end of the vocab
    2. model.resize_token_embeddings(len(tokenizer)) - WITHOUT this the new ids are out of range of the
       embedding matrix (IndexError, or silent garbage on some devices)
    3. the new rows start at the MEAN of the existing embeddings (plus tiny noise so the rows are not
       identical) instead of random noise - they begin as an "average token", which converges faster
       on a small dataset. GPT-2 ties input and output embeddings, so this also initialises the
       output logits for the labels.

    Returns the ids of the category tokens, in LABELS order. Safe to call twice.
    """
    import torch

    tokenizer.add_special_tokens({"additional_special_tokens": LABEL_TOKENS})
    old_size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) != old_size:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            new_rows = [i for i in range(old_size, emb.shape[0])]
            if new_rows:
                mean = emb[:old_size].mean(dim=0)
                emb[new_rows] = mean + init_noise * torch.randn(len(new_rows), emb.shape[1], device=emb.device)
    ids = category_token_ids(tokenizer)
    assert max(ids) < model.get_input_embeddings().num_embeddings, "label id outside embedding matrix"
    return ids


def category_token_ids(tokenizer) -> list[int]:
    """The id of each category token, in LABELS order; each must be a single token."""
    ids = []
    for tok in LABEL_TOKENS:
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{tok} encodes to {len(encoded)} ids; add_category_tokens() not called?")
        ids.append(encoded[0])
    return ids
