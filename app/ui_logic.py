"""The Streamlit logic layer: pure functions, importable and testable without launching Streamlit."""
from __future__ import annotations

from app.dataset import CATEGORIES

DISPLAY_NAMES = {slug: slug.replace("_", " ").title() for slug in CATEGORIES}

EXAMPLES = [
    "The zip on my jacket came apart the first time I wore it.",
    "Anita at the counter was lovely, she found exactly what I needed and even gift wrapped it.",
    "The man at the returns desk rolled his eyes and told me it was my own fault.",
    "Why does a basic notebook cost this much? The shop next door sells it for half.",
    "Only one checkout open on a Saturday and the whole place was packed and stuffy.",
]


def analyze_comment(text: str, classifier) -> dict:
    """Classify `text` and shape the result for display: best category plus all scores, highest first."""
    pred = classifier.classify(text)
    ranked = sorted(pred.scores.items(), key=lambda kv: kv[1], reverse=True)
    return {"label": pred.label, "display": DISPLAY_NAMES[pred.label], "confidence": pred.confidence,
            "ranked": [(DISPLAY_NAMES[k], v) for k, v in ranked],
            "description": CATEGORIES[pred.label]}
