"""``web/public/data/datasets.json``: what the three datasets look like, measured from the processed splits.

Per dataset and class: image counts in the train / validation / test split, the mean image of the training
images, and a handful of training images (random but seeded). Pixels are the original 0-255 grey levels
(recovered from the standardised arrays and rounded), packed as base64 of 784 bytes per image, so the page can
draw them exactly. Nothing here is typed by hand.
"""

import base64
import json

import numpy as np

from src.preprocessing.preprocess import DATASETS, load_dataset

SCHEMA = "writerhand.datasets/1"
N_SAMPLES = 8
SEED = 7


def _raw(X, mean, std):
    """Standardised float32 -> uint8 grey levels (the inverse of ``(x/255 - mean) / (std + 1e-8)``)."""
    return np.clip(np.rint((X * (std + 1e-8) + mean) * 255.0), 0, 255).astype(np.uint8)


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def build_one(name):
    data, meta = load_dataset(name=name)
    y = {s: data[f"y_{s}"].astype(np.int64) for s in ("train", "val", "test")}
    Xtr = _raw(data["X_train"], meta["mean"], meta["std"])
    rng = np.random.default_rng(SEED)
    classes = []
    for k, label in enumerate(DATASETS[name]["classes"]):
        idx = np.flatnonzero(y["train"] == k)
        pick = rng.choice(idx, N_SAMPLES, replace=False)
        classes.append(
            {
                "label": label,
                "n": {s: int((y[s] == k).sum()) for s in y},
                "mean": _b64(np.rint(Xtr[idx].astype(np.float64).mean(0))),
                "samples": [_b64(Xtr[i]) for i in pick],
            }
        )
    ink = (Xtr > 0).mean()
    return {
        "title": DATASETS[name]["title"],
        "kind": DATASETS[name]["kind"],
        "n": {s: int(len(y[s])) for s in y},
        "ink_fraction": float(ink),
        "classes": classes,
    }


def build(names=None):
    names = names or list(DATASETS)
    return {
        "schema": SCHEMA,
        "size": 28,
        "pixels": "uint8, row-major, base64",
        "datasets": {n: build_one(n) for n in names},
    }


def write(path, names=None):
    out = build(names)
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    return out
