"""Training replay: genuine checkpoints and metrics of a real training run.

``record_replay`` re-trains a model with exactly its final configuration and seed while a callback
saves, at selected epochs, (a) a small set of weights, (b) the model's own predictions on a fixed
set of training images, and (c) the measured metrics. Nothing is simulated or interpolated: the
website replays this file. After training, the re-trained weights are compared with the deployed
ones; ``final_matches_deployed`` records the outcome truthfully (CPU training is deterministic for
a fixed seed and thread count, but a mismatch would be reported as ``false``).

AdaBoost is a sequential ensemble, so its states are exactly the prefixes of the deployed ensemble
(first t trees): its replay is built from the deployed model plus the per-round history that the
final run recorded, with no re-training needed.

Saved slices (to keep files small and to say so in the UI): softmax - all class weights;
MLP - the first-layer weights of the first 16 hidden neurons; CNN - all kernels and kernel biases.
Naive Bayes has no iterative training (closed-form maximum likelihood) and no replay.
"""

import json
from pathlib import Path

import numpy as np

from src.models.base import BaseClassifier
from src.utils.paths import SEED

SCHEMA = "writerhand.replay/1"
N_SAMPLES = 12
MLP_NEURONS = 16


def snapshot_epochs(total, extra=()):
    """0, 1, 2, 3, 5, 8, 12, 20, ... plus the last epoch and any epochs in ``extra``."""
    base = {0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 400}
    return sorted({e for e in base if e <= total} | {total} | {e for e in extra if 0 <= e <= total})


def deployed_epoch_guess(best, n_ep):
    return int(best) if best else n_ep


def pick_samples(y, n=N_SAMPLES, seed=SEED):
    """One training image per class first, then random ones, fixed by the seed."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx = [int(rng.choice(np.flatnonzero(y == c))) for c in classes]
    rest = [i for i in rng.permutation(len(y)) if int(i) not in idx]
    idx += [int(i) for i in rest[: max(0, n - len(idx))]]
    return np.array(idx[:n])


def _round(a, sig=5):
    return [float(f"{v:.{sig}g}") for v in np.asarray(a, dtype=np.float64).ravel()]


def _tensors(kind, model):
    if kind == "softmax":
        return {
            "W": {
                "shape": list(model.W.shape),
                "data": _round(model.W),
                "note": "all class weights W[784,K]",
            }
        }
    if kind == "mlp":
        w = model.W[0].numpy()[:, :MLP_NEURONS]
        return {
            f"W0[:, :{MLP_NEURONS}]": {
                "shape": list(w.shape),
                "data": _round(w),
                "note": f"first-layer weights of the first {MLP_NEURONS} hidden neurons (of {model.W[0].shape[1]})",
            }
        }
    if kind == "cnn":
        out = {}
        for i, k in enumerate(model.K):
            out[f"K{i}"] = {
                "shape": list(k.shape),
                "data": _round(k.numpy()),
                "note": f"all kernels of conv {i}",
            }
        return out
    return {}


def _weights(kind, model):
    if kind == "softmax":
        return [model.W, model.b]
    if kind == "mlp":
        return [w.numpy() for w in model.W] + [b.numpy() for b in model.b]
    return [t.numpy() for t in model._params()]


def record_replay(
    kind, model_cls, params, data, meta, deployed_path, dataset, log=print, fit_xy=None
):
    """Re-train ``model_cls(**params)`` with a snapshot callback; returns the replay dictionary."""
    mean, std = meta["mean"], meta["std"]
    idx = pick_samples(data["y_train"])
    Xs = data["X_train"][idx]
    pixels = np.clip(np.rint((Xs * (std + 1e-8) + mean) * 255), 0, 255).astype(int)
    model = model_cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in params.items()})
    snaps, raw = {}, {}

    def cb(epoch, m):
        # every epoch is kept in memory; the file keeps a selection (below)
        hist = m.history or {}
        metrics = {
            k: hist[k][-1] for k in ("loss", "val_acc", "val_loss", "inner_val_loss") if hist.get(k)
        }
        snaps[epoch] = {
            "epoch": epoch,
            "metrics": {k: round(float(v), 5) for k, v in metrics.items()},
            "proba": [_round(r, 5) for r in m.predict_proba(Xs)],
            "tensors": _tensors(kind, m),
        }
        raw[epoch] = [np.array(w, copy=True) for w in _weights(kind, m)]  # exact copy for the check

    Xf, yf = fit_xy if fit_xy is not None else (data["X_train"], data["y_train"])
    model.fit(Xf, yf, data["X_val"], data["y_val"], callback=cb)
    hist = model.history
    n_ep = len(hist["loss"])
    # epochs actually trained may be fewer than the maximum: keep only the snapshots that exist
    best = getattr(model, "best_epoch_", None)
    keep = set(
        snapshot_epochs(n_ep, extra=(best,) if best else ())
    )  # the deployed state is the best epoch
    snapshots = [snaps[e] for e in sorted(snaps) if e in keep]
    # the deployed model is the restored best-epoch state (or the last epoch without early stopping)
    deployed_epoch = int(best) if best else n_ep
    deployed = BaseClassifier.load(deployed_path)
    same = deployed_epoch in raw and all(
        np.array_equal(a, np.asarray(b))
        for a, b in zip(raw[deployed_epoch], _weights(kind, deployed), strict=True)
    )
    series = {k: _round(v, 6) for k, v in hist.items() if k not in ("lr",) and len(v) == n_ep}
    return {
        "schema": SCHEMA,
        "dataset": dataset,
        "model": kind,
        "seed": int(params.get("seed", SEED)),
        "params": params,
        "epochs": list(range(1, n_ep + 1)),
        "series": series,
        "best_epoch": int(best) if best else None,
        "stopped_epoch": int(getattr(model, "stopped_epoch_", n_ep)),
        "samples": {
            "indices": [int(i) for i in idx],
            "labels": [int(v) for v in data["y_train"][idx]],
            "pixels": pixels.tolist(),
        },
        "snapshots": snapshots,
        "deployed_epoch": deployed_epoch,
        "final_matches_deployed": bool(same),
        "note": (
            "Recorded while re-training with the final configuration and seed; snapshots hold the "
            "weights as they were at the end of the listed epochs (before the early-stopping restore); "
            "the deployed model equals the snapshot of `deployed_epoch`. "
            "Saved tensors are slices of the model, see each tensor's note."
        ),
    }


def adaboost_replay(model, history, data, meta, dataset, params):
    """Replay of the deployed ensemble: snapshots are its prefixes (first t trees)."""
    mean, std = meta["mean"], meta["std"]
    idx = pick_samples(data["y_train"])
    Xs = data["X_train"][idx]
    pixels = np.clip(np.rint((Xs * (std + 1e-8) + mean) * 255), 0, 255).astype(int)
    T = model.n_estimators_
    snaps = []
    for t in snapshot_epochs(T):
        scores = np.zeros((len(Xs), len(model.classes)))
        for r in range(t):
            leaf = _leaf_classes(model, r, Xs)
            scores[np.arange(len(Xs)), leaf] += model.alphas[r]
        total = scores.sum(1, keepdims=True)
        proba = scores / np.where(total > 0, total, 1.0)
        m = {}
        if t > 0:
            for key, hk in (("val_acc", "val_acc"), ("loss", "train_err_weighted")):
                if hk in history and len(history[hk]) >= t:
                    m[key] = round(float(history[hk][t - 1]), 5)
        snaps.append(
            {"epoch": t, "metrics": m, "proba": [_round(r, 5) for r in proba], "tensors": {}}
        )
    series = {k: _round(v, 6) for k, v in history.items() if len(v) == T}
    return {
        "schema": SCHEMA,
        "dataset": dataset,
        "model": "adaboost",
        "seed": int(params.get("seed", SEED)),
        "params": params,
        "epochs": list(range(1, T + 1)),
        "series": series,
        "best_epoch": T,
        "stopped_epoch": T,
        "samples": {
            "indices": [int(i) for i in idx],
            "labels": [int(v) for v in data["y_train"][idx]],
            "pixels": pixels.tolist(),
        },
        "snapshots": snaps,
        "deployed_epoch": T,
        "final_matches_deployed": True,
        "note": (
            "AdaBoost adds one tree per round, so the state after round t is exactly the first t "
            "trees of the deployed ensemble; series come from the final training run. Predictions "
            "are weighted vote shares, not calibrated probabilities."
        ),
    }


def _leaf_classes(model, t, X):
    """Class predicted by tree ``t`` for each row of ``X`` (uses the exported array layout)."""
    f, thr, l_, r_, c = (
        model.tree_feature[t],
        model.tree_threshold[t],
        model.tree_left[t],
        model.tree_right[t],
        model.tree_leaf[t],
    )
    node = np.zeros(len(X), dtype=np.int64)
    active = f[node] >= 0
    while active.any():
        i = np.flatnonzero(active)
        go_left = X[i, f[node[i]]] < thr[node[i]]
        node[i] = np.where(go_left, l_[node[i]], r_[node[i]])
        active = f[node] >= 0
    return c[node]


def save_replay(replay, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
    return path
