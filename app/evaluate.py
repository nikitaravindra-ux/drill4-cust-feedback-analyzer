"""Evaluation metrics, written by hand (no sklearn): accuracy, per-class P/R/F1, confusion matrix, baseline."""
from __future__ import annotations

from collections import Counter

from app.dataset import LABELS, Example


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str] = LABELS) -> list[list[int]]:
    """cm[i][j] = how many examples of true class i were predicted as class j."""
    index = {label: i for i, label in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        cm[index[t]][index[p]] += 1
    return cm


def accuracy(cm: list[list[int]]) -> float:
    total = sum(map(sum, cm))
    return sum(cm[i][i] for i in range(len(cm))) / total if total else 0.0


def per_class_metrics(cm: list[list[int]], labels: list[str] = LABELS) -> dict[str, dict[str, float]]:
    out = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        predicted = sum(cm[r][i] for r in range(len(cm)))  # column sum
        actual = sum(cm[i])                                # row sum
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    return out


def macro_f1(metrics: dict[str, dict[str, float]]) -> float:
    return sum(m["f1"] for m in metrics.values()) / len(metrics)


def majority_baseline(train: list[Example], test: list[Example]) -> dict:
    """Always predict the most common training label; report its accuracy on the test set."""
    majority = Counter(ex.label for ex in train).most_common(1)[0][0]
    hits = sum(ex.label == majority for ex in test)
    return {"label": majority, "accuracy": hits / len(test)}


def format_confusion(cm: list[list[int]], labels: list[str] = LABELS) -> str:
    """A markdown table: rows = true class, columns = predicted class."""
    header = "| true \\ predicted | " + " | ".join(labels) + " |"
    rule = "|---|" + "---:|" * len(labels)
    rows = [f"| **{label}** | " + " | ".join(str(n) for n in cm[i]) + " |" for i, label in enumerate(labels)]
    return "\n".join([header, rule, *rows])


def format_report(result: dict) -> str:
    lines = ["| class | precision | recall | F1 | support |", "|---|---:|---:|---:|---:|"]
    for label, m in result["per_class"].items():
        lines.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    lines.append(f"| **macro avg** | | | {result['macro_f1']:.3f} | {result['n']} |")
    return "\n".join(lines)


def evaluate_classifier(classify_fn, test: list[Example], train: list[Example] | None = None) -> dict:
    """Run `classify_fn(text) -> Prediction` over the test split and collect every metric."""
    preds = [classify_fn(ex.text) for ex in test]
    y_true, y_pred = [ex.label for ex in test], [p.label for p in preds]
    cm = confusion_matrix(y_true, y_pred)
    metrics = per_class_metrics(cm)
    result = {"n": len(test), "accuracy": accuracy(cm), "macro_f1": macro_f1(metrics), "per_class": metrics,
              "confusion_matrix": cm, "labels": LABELS,
              "errors": [{"text": ex.text, "true": ex.label, "predicted": p.label, "confidence": p.confidence}
                         for ex, p in zip(test, preds) if ex.label != p.label]}
    if train:
        result["baseline"] = majority_baseline(train, test)
    return result
