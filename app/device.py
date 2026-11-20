"""Pick a torch device: CFA_DEVICE override, else Apple `mps` when available, else CPU."""
import os


def pick_device(override: str | None = None) -> str:
    import torch
    choice = (override or os.environ.get("CFA_DEVICE", "")).strip().lower()
    if choice:
        return choice
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
