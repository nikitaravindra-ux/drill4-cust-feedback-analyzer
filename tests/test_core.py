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
    # default init="words": each new row is the mean embedding of the label's words ("staff praise")
    word_ids = [stub_tokenizer.convert_tokens_to_ids(w) for w in ("staff", "praise")]
    assert torch.allclose(emb.weight[old + LABELS.index("staff_praise")], old_weights[word_ids].mean(dim=0), atol=1e-6)
    toy_model(torch.tensor([ids]))  # would raise IndexError without the resize
    assert core.milestone_2(stub_tokenizer, toy_model) == ids  # idempotent
    assert emb.num_embeddings == old + len(LABELS)


# Milestone 3: Trainer run reduces loss
def test_milestone_3(stub_tokenizer, toy_model, tmp_path):
    from app.finetune import build_trainer, mean_loss, training_arguments
    add_category_tokens(stub_tokenizer, toy_model)
    examples = [Example("the staff was rude", "staff_complaint"), Example("great and very helpful staff", "staff_praise"),
                Example("price too high", "pricing_value"), Example("the store is crowded queue", "store_experience"),
                Example("it arrived broken", "product_issue")] * 4
    args = training_arguments(str(tmp_path), device="cpu", max_steps=25, batch_size=4, lr=5e-3)
    trainer = build_trainer(toy_model, stub_tokenizer, examples, args)
    before = mean_loss(trainer, stub_tokenizer)
    trainer.train()
    after = mean_loss(trainer, stub_tokenizer)
    assert after < before * 0.8, (before, after)


def test_mean_init_sets_new_rows_to_the_average_embedding(stub_tokenizer, toy_model):
    old = toy_model.get_input_embeddings().num_embeddings
    mean = toy_model.get_input_embeddings().weight.detach().mean(dim=0)
    ids = add_category_tokens(stub_tokenizer, toy_model, init="mean")
    new = toy_model.get_input_embeddings().weight[old:]
    assert len(ids) == len(LABELS)
    assert torch.allclose(new, mean.expand(len(ids), -1), atol=0.01)


# Milestone 4: inference scores the category tokens
def test_milestone_4(stub_tokenizer, toy_model):
    from app.classify import Classifier
    core.milestone_2(stub_tokenizer, toy_model)
    clf = Classifier(toy_model, stub_tokenizer, max_length=32)  # toy model has 64 positions
    for text in ["the staff was rude", "price too high", "completely unknown words zzz", "x " * 500]:
        pred = core.milestone_4(text, clf)
        assert pred.label in LABELS  # never anything but a defined category
        assert set(pred.scores) == set(LABELS)
        assert abs(sum(pred.scores.values()) - 1.0) < 1e-5
        assert pred.label == max(pred.scores, key=pred.scores.get)
        assert core.milestone_4(text, clf) == pred  # deterministic
    import pytest
    with pytest.raises(ValueError):
        clf.classify("   ")


def test_loading_a_missing_model_gives_a_clear_error(tmp_path):
    import pytest
    from app.classify import ModelNotFoundError, load_classifier
    with pytest.raises(ModelNotFoundError, match="python -m app.cli train"):
        load_classifier(tmp_path / "nope")
