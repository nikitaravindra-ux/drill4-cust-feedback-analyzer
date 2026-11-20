"""Orchestration layer for the Customer Feedback Analyzer: one function per milestone.

The project fine-tunes a small GPT-2 (``distilgpt2``) to tag customer comments with one of five new special
category tokens. Each ``milestone_N`` below calls the module that implements it, adds the orchestration around
it, and returns a result dict (or, for milestone 1, the stats dict) that can be printed as-is:

    milestone_1  app/dataset.py   labelled dataset + stratified split
    milestone_2  app/tokens.py    ``add_special_tokens`` + ``resize_token_embeddings`` + mean-initialised rows
    milestone_3  app/finetune.py  Hugging Face ``Trainer`` + ``TrainingArguments``
    milestone_4  app/classify.py  inference by scoring the category-token logits
    milestone_5  app/ui_logic.py  logic behind the Streamlit page (app/main.py)

Every milestone can be run with no arguments, e.g. ``python -m app.cli milestone 2``.
"""
from __future__ import annotations

from app.ui_logic import EXAMPLES

STREAMLIT_COMMAND = "streamlit run app/main.py"


def milestone_1() -> dict:
    """Build the labelled dataset (app/dataset.py): five categories, feedback.csv, stratified train/test split.

    Returns the dataset stats, including the overall, train and test class balance.
    """
    from app.dataset import build_dataset
    stats = build_dataset()
    stats["categories"] = list(stats["class_balance"])
    total = stats["total"]
    stats["class_share"] = {k: round(v / total, 3) for k, v in stats["class_balance"].items()}
    return stats


def milestone_2(tokenizer=None, model=None, base: str = "distilgpt2") -> dict:
    """Extend the tokenizer AND the model with the category tokens (app/tokens.py).

    Sequence: ``tokenizer.add_special_tokens`` -> ``model.resize_token_embeddings(len(tokenizer))`` -> the new
    embedding rows are initialised from existing embeddings (mean of the label's words) rather than left random.
    Without ``resize_token_embeddings`` the new ids would fall outside the embedding matrix.

    Loads ``base`` from the Hugging Face Hub when no tokenizer/model is supplied. Returns the token ids, the
    vocab size before and after, and whether the embedding matrix was resized to match the tokenizer.
    """
    from app.tokens import LABEL_TOKENS, add_category_tokens
    if tokenizer is None or model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(base)
        model = model or AutoModelForCausalLM.from_pretrained(base)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    vocab_before = len(tokenizer)
    embeddings_before = model.get_input_embeddings().num_embeddings
    ids = add_category_tokens(tokenizer, model)
    embeddings_after = model.get_input_embeddings().num_embeddings
    return {"tokens": dict(zip(LABEL_TOKENS, ids)), "token_ids": ids,
            "vocab_size_before": vocab_before, "vocab_size_after": len(tokenizer),
            "embedding_rows_before": embeddings_before, "embedding_rows_after": embeddings_after,
            "embeddings_resized_to_match": embeddings_after == len(tokenizer)}


def milestone_3(**kwargs) -> dict:
    """Fine-tune with the Hugging Face ``Trainer`` (app/finetune.py) and save the model to models/.

    ``Trainer`` is built from ``TrainingArguments`` (see ``training_arguments`` in app/finetune.py); the loss is
    computed only on the category token. Keyword arguments (base, epochs, batch_size, lr, device, seed) go to
    ``train``. Returns the summary: TrainingArguments used, epochs, final training loss, where it was saved.
    """
    from app.finetune import train
    info = train(**kwargs)
    return {k: v for k, v in info.items() if k != "loss_history"} | {
        "first_logged_loss": info["loss_history"][0] if info["loss_history"] else None,
        "last_logged_loss": info["loss_history"][-1] if info["loss_history"] else None}


def milestone_4(text: str = EXAMPLES[0], classifier=None) -> dict:
    """Classify one comment (app/classify.py): one forward pass, softmax over the category-token logits.

    Returns the label, its confidence and the full score distribution across every category.
    """
    from app.classify import load_classifier
    pred = (classifier or load_classifier()).classify(text)
    return {"text": text, "label": pred.label, "confidence": pred.confidence,
            "scores": dict(sorted(pred.scores.items(), key=lambda kv: -kv[1]))}


def milestone_5(text: str = EXAMPLES[1], classifier=None) -> dict:
    """The Streamlit logic layer (app/ui_logic.py; the page itself is app/main.py).

    Returns the label, all scores (highest first), the display fields the page renders, and the command that
    launches the Streamlit page.
    """
    from app.classify import load_classifier
    from app.ui_logic import analyze_comment
    classifier = classifier or load_classifier()
    result = analyze_comment(text, classifier)
    scores = classifier.classify(text).scores
    return {**result, "text": text, "scores": dict(sorted(scores.items(), key=lambda kv: -kv[1])),
            "launch_command": STREAMLIT_COMMAND}
