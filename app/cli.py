"""Headless interface:  python -m app.cli {dataset,train,classify,evaluate} ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.dataset import ROOT

RESULTS = ROOT / "results" / "evaluation.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dataset", help="validate the raw comments, write feedback.csv + train/test split")
    t = sub.add_parser("train", help="fine-tune GPT-2 with the Trainer and save to models/feedback-gpt2")
    t.add_argument("--base", default="distilgpt2", help="distilgpt2 (default) or gpt2")
    t.add_argument("--epochs", type=float, default=8)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--device", default=None, help="mps / cpu / cuda (default: auto, or $CFA_DEVICE)")
    c = sub.add_parser("classify", help="classify one comment")
    c.add_argument("text")
    c.add_argument("--device", default=None)
    e = sub.add_parser("evaluate", help="evaluate on the held-out test split")
    e.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    from app import core
    if args.command == "dataset":
        print(json.dumps(core.milestone_1(), indent=2))
    elif args.command == "train":
        info = core.milestone_3(base=args.base, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                device=args.device)
        print(json.dumps({k: v for k, v in info.items() if k != "loss_history"}, indent=2))
    else:
        from app.classify import ModelNotFoundError, load_classifier
        try:
            clf = load_classifier(device=args.device)
        except ModelNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        if args.command == "classify":
            pred = clf.classify(args.text)
            print(f"{pred.label}  ({pred.confidence:.1%})")
            for label, p in sorted(pred.scores.items(), key=lambda kv: -kv[1]):
                print(f"  {label:<18}{p:.3f}")
        else:
            from app.dataset import load_split
            from app.evaluate import evaluate_classifier, format_confusion, format_report
            result = evaluate_classifier(clf.classify, load_split("test"), load_split("train"))
            RESULTS.parent.mkdir(exist_ok=True)
            RESULTS.write_text(json.dumps(result, indent=2))
            print(f"accuracy {result['accuracy']:.3f} on {result['n']} held-out comments "
                  f"(majority baseline: {result['baseline']['accuracy']:.3f} by always saying "
                  f"{result['baseline']['label']})\n")
            print(format_report(result), "\n")
            print(format_confusion(result["confusion_matrix"]))
            print(f"\n{len(result['errors'])} errors, written to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
