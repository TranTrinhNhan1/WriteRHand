"""Optional study: does data augmentation help the MLP?  (controlled, multi-seed)

Augmented copies are generated from the **training split only** (random rotation, scaling and
translation; labels unchanged) with ``scipy.ndimage`` - the augmentation is a data transform, not
a model component. Validation / test images are never augmented. Everything except the training
set (hyper-parameters, seeds, early stopping rule) is held fixed between the two conditions.
"""

import json
import time

import numpy as np
from scipy.ndimage import affine_transform

from src.evaluation.classification_metrics import get_metrics
from src.models.mlp import MLP
from src.utils.paths import RESULTS_DIR


def augment_images(X, k=1, seed=0, max_rot_deg=10.0, max_shift=2.0, scale=(0.9, 1.1)):
    """Return ``k`` randomly transformed copies of the rows of ``X`` (N, 784), stacked (k*N, 784).

    The background value (smallest value of ``X``) fills the area uncovered by the transform.
    """
    X = np.asarray(X)
    n = len(X)
    bg = float(X.min())
    rng = np.random.default_rng(seed)
    out = np.empty((k * n, X.shape[1]), dtype=X.dtype)
    centre = np.array([13.5, 13.5])
    for rep in range(k):
        ang = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg, n))
        sc = rng.uniform(scale[0], scale[1], n)
        shift = rng.uniform(-max_shift, max_shift, (n, 2))
        for i in range(n):
            c, s = np.cos(ang[i]), np.sin(ang[i])
            # output->input mapping: rotate/scale about the centre, then translate
            M = np.array([[c, s], [-s, c]]) / sc[i]
            offset = centre - M @ centre - shift[i]
            img = affine_transform(
                X[i].reshape(28, 28), M, offset=offset, order=1, mode="constant", cval=bg
            )
            out[rep * n + i] = img.reshape(-1)
    return out


def run_study(data, params, seeds=(42, 43, 44), k=2, aug_seed=123, log=print):
    results = {"params": params, "k_copies": k, "seeds": list(seeds), "conditions": {}}
    X, y = data["X_train"], data["y_train"]
    t0 = time.perf_counter()
    X_aug = augment_images(X, k=k, seed=aug_seed)
    results["augmentation_time_sec"] = time.perf_counter() - t0
    Xa = np.vstack([X, X_aug])
    ya = np.concatenate([y, np.tile(y, k)])
    conds = {"baseline": (X, y), f"augmented_x{k + 1}": (Xa, ya)}
    for name, (Xc, yc) in conds.items():
        rows = []
        for seed in seeds:
            p = {**params, "seed": seed, "hidden_sizes": tuple(params["hidden_sizes"])}
            t1 = time.perf_counter()
            m = MLP(**p).fit(Xc, yc)
            fit = time.perf_counter() - t1
            row = {
                "seed": seed,
                "n_train": int(len(yc)),
                "fit_time_sec": fit,
                "stopped_epoch": int(m.stopped_epoch_),
            }
            for split in ("train", "val", "test"):
                row[f"{split}_acc"] = get_metrics(
                    data[f"y_{split}"], m.predict(data[f"X_{split}"]), labels=range(10)
                )["accuracy"]
            rows.append(row)
            log(
                f"[augment:{name}] seed={seed} n={len(yc)} val={row['val_acc']:.4f} "
                f"test={row['test_acc']:.4f} fit={fit:.0f}s"
            )
        accs = {s: np.array([r[f"{s}_acc"] for r in rows]) for s in ("train", "val", "test")}
        results["conditions"][name] = {
            "runs": rows,
            **{f"{s}_acc_mean": float(a.mean()) for s, a in accs.items()},
            **{
                f"{s}_acc_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0 for s, a in accs.items()
            },
        }
    return results


def save_study(results, path=None):
    path = path or RESULTS_DIR / "augmentation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return path
