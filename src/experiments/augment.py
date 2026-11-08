"""Data augmentation: five transformations and a controlled ablation study.

Transformations (all act on the *training split only*; labels never change):

==================  ================================================================  =============
name                effect                                                            default range
==================  ================================================================  =============
``rotation``        rotate about the image centre by U(-a, a) degrees                  a = 15
``translation``     shift by U(-s, s) pixels along each axis                           s = 2
``scaling``         zoom about the centre by a factor U(lo, hi)                        (0.9, 1.1)
``gaussian_noise``  add N(0, sigma^2) to pixel intensities in [0, 1], then clip         sigma = 0.1
``salt_pepper``     a fraction q of the pixels is set to 0 or 1 with equal chance       q = 0.02
==================  ================================================================  =============

Geometric transformations are composed into ONE affine map per image (``scipy.ndimage`` with
bilinear interpolation: a data transform, not a model component); noise is applied afterwards in
pixel units. The pixel scale is recovered from the data itself (the minimum is the background, the
maximum is full ink), so the functions work on standardised features.

Ablation protocol (``run_ablation``): everything but the training set is fixed (hyper-parameters,
seeds, early-stopping rule). Each condition adds ``k`` randomly transformed copies of every training
image. Conditions are compared on the **validation** split (mean +- std over seeds); the test split
is reported for information and for a fixed robustness benchmark, never used to choose a condition.
"""

import json
import time

import numpy as np
from scipy.ndimage import affine_transform

from src.evaluation.classification_metrics import get_metrics
from src.models.cnn import CNN
from src.models.mlp import MLP
from src.utils.paths import RESULTS_DIR

TRANSFORMS = ("rotation", "translation", "scaling", "gaussian_noise", "salt_pepper")
GEOMETRIC = ("rotation", "translation", "scaling")
DEFAULTS = {
    "rotation": 15.0,
    "translation": 2.0,
    "scaling": (0.9, 1.1),
    "gaussian_noise": 0.1,
    "salt_pepper": 0.02,
}
# fixed, stronger stress levels used only to measure robustness of the trained models
STRESS = {
    "rotation": 20.0,
    "translation": 3.0,
    "scaling": (0.8, 1.2),
    "gaussian_noise": 0.2,
    "salt_pepper": 0.05,
}
CONDITIONS = {
    "baseline": (),
    "rotation": ("rotation",),
    "translation": ("translation",),
    "scaling": ("scaling",),
    "gaussian_noise": ("gaussian_noise",),
    "salt_pepper": ("salt_pepper",),
    "geometric": GEOMETRIC,
    "all": TRANSFORMS,
}


def _pixel_scale(X):
    bg, hi = float(X.min()), float(X.max())
    return bg, (hi - bg) if hi > bg else 1.0


def transform_images(X, ops, k=1, seed=0, **ranges):
    """Return ``k`` randomly transformed copies of the rows of ``X`` (N, 784) stacked (k*N, 784).

    ``ops`` is a subset of ``TRANSFORMS``; ``ranges`` overrides ``DEFAULTS`` (e.g. ``rotation=25``).
    """
    unknown = set(ops) - set(TRANSFORMS)
    if unknown:
        raise ValueError(f"unknown transformations: {sorted(unknown)}")
    cfg = {**DEFAULTS, **ranges}
    X = np.asarray(X)
    n = len(X)
    bg, span = _pixel_scale(X)
    rng = np.random.default_rng(seed)
    out = np.empty((k * n, X.shape[1]), dtype=X.dtype)
    centre = np.array([13.5, 13.5])
    geo = [o for o in GEOMETRIC if o in ops]
    for rep in range(k):
        # draws happen in a fixed order so that conditions with the same seed are comparable
        ang = np.deg2rad(rng.uniform(-cfg["rotation"], cfg["rotation"], n))
        sc = rng.uniform(cfg["scaling"][0], cfg["scaling"][1], n)
        shift = rng.uniform(-cfg["translation"], cfg["translation"], (n, 2))
        if "rotation" not in ops:
            ang = np.zeros(n)
        if "scaling" not in ops:
            sc = np.ones(n)
        if "translation" not in ops:
            shift = np.zeros((n, 2))
        gauss = rng.normal(0.0, cfg["gaussian_noise"], X.shape)
        flip = rng.random(X.shape)
        salt = rng.random(X.shape) < 0.5
        block = slice(rep * n, (rep + 1) * n)
        if geo:
            for i in range(n):
                c, s = np.cos(ang[i]), np.sin(ang[i])
                M = np.array([[c, s], [-s, c]]) / sc[i]  # output -> input mapping
                offset = centre - M @ centre - shift[i]
                img = affine_transform(
                    X[i].reshape(28, 28), M, offset=offset, order=1, mode="constant", cval=bg
                )
                out[rep * n + i] = img.reshape(-1)
        else:
            out[block] = X
        if "gaussian_noise" in ops or "salt_pepper" in ops:
            p = (out[block].astype(np.float64) - bg) / span
            if "gaussian_noise" in ops:
                p = np.clip(p + gauss, 0.0, 1.0)
            if "salt_pepper" in ops:
                hit = flip < cfg["salt_pepper"]
                p = np.where(hit, salt.astype(np.float64), p)
            out[block] = (bg + p * span).astype(X.dtype)
    return out


def augment_images(X, k=1, seed=0, max_rot_deg=10.0, max_shift=2.0, scale=(0.9, 1.1)):
    """Geometric augmentation only (rotation + translation + scaling); kept for the first study."""
    return transform_images(
        X, GEOMETRIC, k=k, seed=seed, rotation=max_rot_deg, translation=max_shift, scaling=scale
    )


def perturb_benchmark(X, seed=999):
    """Fixed perturbed copies of ``X`` for the robustness benchmark: ``{name: array}``."""
    return {
        name: transform_images(X, (name,), k=1, seed=seed, **{name: STRESS[name]})
        for name in TRANSFORMS
    }


def example_grid(X, y, n=8, seed=0):
    """``(titles, images)`` rows for the figure: original + each transformation at default and
    stress strength, same source images."""
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    src = X[idx]
    rows = [("original", src)]
    for name in TRANSFORMS:
        rows.append((f"{name} (train range)", transform_images(src, (name,), seed=seed)))
        rows.append(
            (f"{name} (stress)", transform_images(src, (name,), seed=seed, **{name: STRESS[name]}))
        )
    return rows, y[idx]


def make_model(kind, params, seed):
    """``kind``: ``mlp`` | ``cnn`` for the ablation study; any key of ``SPECS`` for the deployed variants."""
    p = {**params, "seed": seed}
    if kind == "mlp":
        return MLP(**{**p, "hidden_sizes": tuple(p["hidden_sizes"])})
    if kind == "cnn":
        return CNN(**{k: tuple(v) if k in ("channels", "dense") else v for k, v in p.items()})
    from src.experiments.search import SPECS

    if kind in SPECS:
        cls = SPECS[kind]["cls"]
        if kind == "naive_bayes":
            p = {k: v for k, v in params.items() if k != "seed"}
        return cls(**p)
    raise ValueError(kind)


# --------------------------------------------------------------------------- deployed models
# The demo models (MLP, CNN) are trained on the original images plus K augmented copies. The rotation
# range is chosen on the validation split among the candidates below; everything else is fixed.
DEPLOY_COPIES = 2
DEPLOY_SEED = 123
DEPLOY_ROTATIONS = (15.0, 30.0, 45.0)
# fixed geometric stress applied to the validation split to compare candidates (test is not used)
DEPLOY_STRESS = {"rotation": 30.0, "translation": 3.0, "scaling": (0.8, 1.2)}
DEPLOY_MAX_CLEAN_DROP = (
    0.005  # a candidate may not lose more than 0.5 pt of clean validation accuracy
)
ROTATION_ANGLES = (
    0,
    10,
    20,
    30,
    45,
    90,
)  # information only: test images rotated by a random angle in +-a


def deploy_spec(rotation, k=DEPLOY_COPIES):
    """Augmentation of a deployed model: rotation (+-rotation deg), shift (+-2 px), scale (0.9-1.1)."""
    return {
        "ops": list(GEOMETRIC),
        "k": int(k),
        "seed": DEPLOY_SEED,
        "ranges": {"rotation": float(rotation)},
    }


def apply_spec(X, y, spec):
    """Original images followed by ``k`` randomly transformed copies, as described by ``spec``."""
    ranges = {
        k_: tuple(v) if isinstance(v, list) else v for k_, v in (spec.get("ranges") or {}).items()
    }
    Xa = transform_images(X, tuple(spec["ops"]), k=spec["k"], seed=spec["seed"], **ranges)
    return np.vstack([X, Xa]), np.concatenate([y, np.tile(y, spec["k"])])


def _acc(model, X, y):
    return float(np.mean(model.predict(X) == y))


def _probe(model, data, stress_val):
    """Clean / stressed validation accuracy and test accuracy at several rotation levels."""
    return {
        "val_acc": _acc(model, data["X_val"], data["y_val"]),
        "val_stress_acc": _acc(model, stress_val, data["y_val"]),
        "test_acc": _acc(model, data["X_test"], data["y_test"]),
        "test_rotation_acc": {
            str(a): _acc(
                model,
                data["X_test"]
                if a == 0
                else transform_images(
                    data["X_test"], ("rotation",), k=1, seed=999, rotation=float(a)
                ),
                data["y_test"],
            )
            for a in ROTATION_ANGLES
        },
    }


def select_deployed(data, kind, params, baseline_model, log=print, rotations=DEPLOY_ROTATIONS):
    """Choose the rotation range of the deployed ``kind`` on validation only.

    Rule: among the candidates whose clean validation accuracy is within ``DEPLOY_MAX_CLEAN_DROP`` of the
    non-augmented model, take the highest accuracy on the geometrically stressed validation split. The test
    split is evaluated for information (rotation table) and never used to choose.
    """
    stress_val = transform_images(data["X_val"], GEOMETRIC, k=1, seed=999, **DEPLOY_STRESS)
    out = {
        "schema": 1,
        "model": kind,
        "params": params,
        "copies": DEPLOY_COPIES,
        "stress_on_validation": {
            k_: list(v) if isinstance(v, tuple) else v for k_, v in DEPLOY_STRESS.items()
        },
        "max_clean_val_drop": DEPLOY_MAX_CLEAN_DROP,
        "rule": "highest stressed-validation accuracy among candidates whose clean validation accuracy "
        "is within max_clean_val_drop of the non-augmented model; test is never used to choose",
        "baseline": _probe(baseline_model, data, stress_val),
        "candidates": {},
    }
    log(
        f"[augment-select:{kind}] baseline {out['baseline']['val_acc']:.4f} / stress {out['baseline']['val_stress_acc']:.4f}"
    )
    for rot in rotations:
        spec = deploy_spec(rot)
        t0 = time.perf_counter()
        Xf, yf = apply_spec(data["X_train"], data["y_train"], spec)
        m = make_model(kind, params, params.get("seed", 42))
        if kind == "naive_bayes":
            m.fit(Xf, yf)
        else:
            m.fit(Xf, yf, data["X_val"], data["y_val"])
        row = {
            "spec": spec,
            "fit_time_sec": time.perf_counter() - t0,
            "n_train": int(len(yf)),
            "stopped_epoch": int(getattr(m, "stopped_epoch_", 0) or 0),
            **_probe(m, data, stress_val),
        }
        out["candidates"][f"rotation_{int(rot)}"] = row
        log(
            f"[augment-select:{kind}] rot={rot:.0f} val={row['val_acc']:.4f} "
            f"stress={row['val_stress_acc']:.4f} fit={row['fit_time_sec']:.0f}s"
        )
    floor = out["baseline"]["val_acc"] - DEPLOY_MAX_CLEAN_DROP
    ok = {k_: v for k_, v in out["candidates"].items() if v["val_acc"] >= floor}
    out["chosen"] = max(ok, key=lambda k_: ok[k_]["val_stress_acc"]) if ok else None
    out["spec"] = out["candidates"][out["chosen"]]["spec"] if out["chosen"] else None
    return out


def run_ablation(
    data, kind, params, seeds=(42, 43, 44), k=1, conditions=None, aug_seed=123, log=print
):
    """Train ``kind`` (``mlp`` | ``cnn``) once per condition and seed; see the module docstring."""
    conditions = conditions or CONDITIONS
    X, y = data["X_train"], data["y_train"]
    stress = perturb_benchmark(data["X_test"])
    results = {
        "schema": 2,
        "model": kind,
        "params": params,
        "k_copies": k,
        "seeds": list(seeds),
        "defaults": {k_: list(v) if isinstance(v, tuple) else v for k_, v in DEFAULTS.items()},
        "stress": {k_: list(v) if isinstance(v, tuple) else v for k_, v in STRESS.items()},
        "selection_rule": "mean validation accuracy over seeds; test is reported, never used to choose",
        "conditions": {},
    }
    for name, ops in conditions.items():
        t0 = time.perf_counter()
        if ops:
            Xc = np.vstack([X, transform_images(X, ops, k=k, seed=aug_seed)])
            yc = np.concatenate([y, np.tile(y, k)])
        else:
            Xc, yc = X, y
        t_aug = time.perf_counter() - t0
        rows = []
        for seed in seeds:
            t1 = time.perf_counter()
            m = make_model(kind, params, seed).fit(Xc, yc)
            fit = time.perf_counter() - t1
            row = {
                "seed": seed,
                "n_train": int(len(yc)),
                "fit_time_sec": fit,
                "stopped_epoch": int(m.stopped_epoch_),
            }
            for split in ("train", "val", "test"):
                met = get_metrics(
                    data[f"y_{split}"], m.predict(data[f"X_{split}"]), labels=range(10)
                )
                row[f"{split}_acc"] = met["accuracy"]
                if split != "train":
                    row[f"{split}_f1_macro"] = met["f1_macro"]
            row["robust_test_acc"] = {
                t: float(np.mean(m.predict(Xs) == data["y_test"])) for t, Xs in stress.items()
            }
            rows.append(row)
            log(
                f"[augment:{kind}:{name}] seed={seed} n={len(yc)} val={row['val_acc']:.4f} "
                f"test={row['test_acc']:.4f} fit={fit:.0f}s"
            )
        accs = {s: np.array([r[f"{s}_acc"] for r in rows]) for s in ("train", "val", "test")}
        rob = {t: np.array([r["robust_test_acc"][t] for r in rows]) for t in TRANSFORMS}
        results["conditions"][name] = {
            "ops": list(ops),
            "augmentation_time_sec": t_aug,
            "runs": rows,
            **{f"{s}_acc_mean": float(a.mean()) for s, a in accs.items()},
            **{
                f"{s}_acc_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0 for s, a in accs.items()
            },
            "robust_test_acc_mean": {t: float(a.mean()) for t, a in rob.items()},
        }
    base = results["conditions"].get("baseline")
    if base:
        for c in results["conditions"].values():
            diffs = np.array([r["val_acc"] for r in c["runs"]]) - np.array(
                [r["val_acc"] for r in base["runs"]]
            )
            c["val_delta_vs_baseline_mean"] = float(diffs.mean())
            c["val_delta_vs_baseline_sd"] = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    results["selected_by_validation"] = max(
        results["conditions"], key=lambda n: results["conditions"][n]["val_acc_mean"]
    )
    return results


def save_study(results, path=None):
    path = path or RESULTS_DIR / f"augmentation_{results['model']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return path


def run_study_cli(data, kind, params, seeds, k, conditions, results_dir):
    study = run_ablation(data, kind, params, seeds=seeds, k=k, conditions=conditions)
    return save_study(study, results_dir / f"augmentation_{kind}.json")
