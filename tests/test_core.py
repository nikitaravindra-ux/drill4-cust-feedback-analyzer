"""One real test per milestone. All offline and fast: tiny GPT-2, stub tokenizer, no Streamlit, no downloads."""
import torch

from app import core
from app.dataset import (CATEGORIES, IGNORE_INDEX, LABELS, SEPARATOR, FeedbackDataset, PadCollator, Example,
                         build_dataset, class_balance, encode_example, label_token, load_raw, stratified_split)
from app.tokens import LABEL_TOKENS, add_category_tokens, category_token_ids


# Milestone 1: categories + dataset
def test_milestone_1(tmp_path, stub_tokenizer):
    assert 3 <= len(CATEGORIES) <= 5
    examples = load_raw()
    balance = class_balance(examples)
    assert all(60 <= n <= 100 for n in balance.values()), balance
    assert len({e.text.lower() for e in examples}) == len(examples)  # no duplicates

    stats = build_dataset(out_dir=tmp_path)
    assert stats["train"] + stats["test"] == len(examples)
    train, test = stratified_split(examples)
    assert not {e.text for e in train} & {e.text for e in test}  # no leakage
    for label in LABELS:  # every class is in the test set, in proportion
        assert class_balance(test)[label] in range(round(balance[label] * 0.2) - 1, round(balance[label] * 0.2) + 2)

    # templated, and only the label token is unmasked
    model_vocab_tokens = stub_tokenizer.add_special_tokens({"additional_special_tokens": LABEL_TOKENS})
    ids = category_token_ids(stub_tokenizer)
    item = encode_example(stub_tokenizer, "the staff was very helpful", ids[1])
    assert item["input_ids"][-2:] == [ids[1], stub_tokenizer.eos_token_id]
    unmasked = [(i, y) for i, y in enumerate(item["labels"]) if y != IGNORE_INDEX]
    assert unmasked == [(len(item["input_ids"]) - 2, ids[1])]
    sep = stub_tokenizer(SEPARATOR, add_special_tokens=False)["input_ids"]
    assert item["input_ids"][-2 - len(sep):-2] == sep
    # padded batch: padding positions are masked too
    batch = PadCollator(stub_tokenizer.pad_token_id)([item, encode_example(stub_tokenizer, "rude", ids[0])])
    assert batch["labels"].shape == batch["input_ids"].shape
    assert (batch["labels"] != IGNORE_INDEX).sum().item() == 2
    assert model_vocab_tokens == len(LABEL_TOKENS)


# Milestone 2: special tokens + embedding resize
def test_milestone_2(stub_tokenizer, toy_model):
    old = toy_model.get_input_embeddings().num_embeddings
    old_weights = toy_model.get_input_embeddings().weight.detach().clone()
    ids = core.milestone_2(stub_tokenizer, toy_model)
    emb = toy_model.get_input_embeddings()
    assert emb.num_embeddings == len(stub_tokenizer) == old + len(LABELS)
    for tok, i in zip(LABEL_TOKENS, ids):
        assert stub_tokenizer.encode(tok, add_special_tokens=False) == [i]  # exactly one id
        assert i < emb.num_embeddings
        assert stub_tokenizer.decode([i]) == tok  # round trip
    assert torch.equal(emb.weight[:old], old_weights)  # old rows untouched
    mean = old_weights.mean(dim=0)
    assert torch.allclose(emb.weight[old:], mean.expand(len(ids), -1), atol=0.02)  # mean-initialised
    toy_model(torch.tensor([ids]))  # would raise IndexError without the resize
    assert core.milestone_2(stub_tokenizer, toy_model) == ids  # idempotent
    assert emb.num_embeddings == old + len(LABELS)
