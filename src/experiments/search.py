"""Hyper-parameter search: sequential one-factor sweeps selected on the validation split.

Why not a grid?  The old notebook defined a 960-configuration x 4-fold MLP grid (3 840 fits).
Here each hyper-parameter is swept in turn while the others stay at the best value found so far
(coordinate descent). This costs sum(len(values)) fits instead of prod(len(values)), and every
sweep is exactly the "change one hyper-parameter at a time" plot the assignment asks for.

* The **validation** split is the only split used to compare configurations.
* The **test** split is never loaded here: ``tune`` only needs ``X_train/y_train/X_val/y_val``
  (``tests/test_experiments.py`` enforces this by running it without test arrays).
* Every evaluated configuration is appended to ``results/tuning/<model>.json``; nothing is
  hidden or edited by hand.
"""

import json
import time

import numpy as np

from src.evaluation.classification_metrics import get_metrics
from src.experiments.env import git_info, hardware_info
from src.models.adaboost import AdaBoostSAMME
from src.models.cnn import CNN
from src.models.logistic_regression import SoftmaxRegression
from src.models.mlp import MLP
from src.models.naive_bayes import GaussianNB
from src.models.svm import LinearSVM
from src.utils.paths import RESULTS_DIR

SEED = 42

SPECS = {
    "naive_bayes": {
        "cls": GaussianNB,
        "base": {"var_smoothing": 1e-9},
        "sweeps": [
            ("var_smoothing", [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0])
        ],
    },
    "softmax": {
        "cls": SoftmaxRegression,
        "base": {"learning_rate": 0.1, "l2": 1e-4, "epochs": 30, "batch_size": 128, "seed": SEED},
        "sweeps": [
            ("learning_rate", [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]),
            ("batch_size", [16, 32, 64, 128, 256, 1024]),
            ("l2", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
        ],
        "epoch_curve": {"param": "epochs", "max": 100},
    },
    "svm": {
        "cls": LinearSVM,
        "base": {
            "learning_rate": 0.01,
            "l2": 1e-4,
            "loss": "hinge",
            "epochs": 30,
            "batch_size": 128,
            "seed": SEED,
        },
        "sweeps": [
            ("loss", ["hinge", "squared_hinge"]),
            ("learning_rate", [0.0003, 0.001, 0.003, 0.01, 0.03]),
            ("l2", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
            ("batch_size", [32, 128, 512]),
        ],
        "epoch_curve": {"param": "epochs", "max": 100},
    },
    "mlp": {
        "cls": MLP,
        "base": {
            "hidden_sizes": (256,),
            "learning_rate": 0.05,
            "momentum": 0.9,
            "p_drop": 0.0,
            "batch_size": 128,
            "epochs": 40,
            "seed": SEED,
        },
        "sweeps": [
            (
                "hidden_sizes",
                [(32,), (64,), (128,), (256,), (512,), (256, 128), (512, 256), (512, 256, 128)],
            ),
            ("learning_rate", [0.005, 0.01, 0.02, 0.05, 0.1]),
            ("momentum", [0.0, 0.5, 0.9, 0.95]),
            ("batch_size", [32, 64, 128, 256]),
            ("p_drop", [0.0, 0.1, 0.2, 0.3, 0.5]),
        ],
    },
    "adaboost": {
        "cls": AdaBoostSAMME,
        # sweeps use 50 rounds (cost); the final number of rounds is read from the staged
        # validation curve of one longer run (epoch_curve below)
        "base": {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 1.0,
            "n_bins": 32,
            "max_features": 1.0,
            "seed": SEED,
        },
        "sweeps": [
            ("max_depth", [2, 4, 6, 8, 10]),
            ("max_features", [0.25, 0.5, 1.0]),
            ("learning_rate", [0.1, 0.3, 0.5, 1.0]),
            ("n_bins", [8, 16, 32, 64]),
        ],
        "epoch_curve": {"param": "n_estimators", "max": 400},
    },
    "cnn": {
        "cls": CNN,
        "base": {
            "channels": (8, 16),
            "kernel_size": 3,
            "dense": (),
            "learning_rate": 0.03,
            "momentum": 0.9,
            "p_drop": 0.0,
            "batch_size": 128,
            "epochs": 20,
            "seed": SEED,
        },
        "sweeps": [
            ("channels", [(4, 8), (8, 16), (16, 32)]),
            ("kernel_size", [3, 5]),
            ("dense", [(), (64,)]),
            ("learning_rate", [0.01, 0.03, 0.1]),
            ("batch_size", [64, 128, 256]),
            ("p_drop", [0.0, 0.2]),
        ],
    },
}
CLASS_NAME = {k: v["cls"].__name__ for k, v in SPECS.items()}
BASELINE_MODELS = [k for k in SPECS if k != "cnn"]  # models with a scikit-learn counterpart

# tiny version used by CI / quick sanity runs (``--smoke``)
SMOKE = {
    "n_train": 2000,
    "n_val": 1000,
    "sweeps": {
        "naive_bayes": [("var_smoothing", [1e-9, 0.1])],
        "softmax": [("learning_rate", [0.03, 0.1])],
        "svm": [("learning_rate", [0.003, 0.01])],
        "mlp": [("hidden_sizes", [(16,), (32,)])],
        "adaboost": [("max_depth", [1, 2])],
        "cnn": [("channels", [(2, 4), (4, 8)])],
    },
    "base_override": {
        "softmax": {"epochs": 3},
        "svm": {"epochs": 3},
        "mlp": {"epochs": 3},
        "adaboost": {"n_estimators": 5},
        "cnn": {"epochs": 2},
    },
    "curve_max": {"softmax": 6, "svm": 6, "adaboost": 10},
}


# Reduced sweeps for the extra datasets (Fashion-MNIST, EMNIST Digits): same coordinate-descent
# protocol, fewer values per hyper-parameter so that the whole study stays affordable on a CPU.
# AdaBoost (the most expensive model) keeps a single sweep and its staged curve.
LIGHT = {
    "naive_bayes": [("var_smoothing", [1e-9, 1e-6, 1e-3, 1e-2, 0.1, 1.0])],
    "softmax": [("learning_rate", [0.03, 0.1, 0.3]), ("l2", [0.0, 1e-4, 1e-3])],
    "svm": [("learning_rate", [0.003, 0.01, 0.03]), ("l2", [1e-5, 1e-4, 1e-3])],
    "mlp": [
        ("hidden_sizes", [(128,), (256,), (512,), (512, 256)]),
        ("learning_rate", [0.02, 0.05, 0.1]),
        ("p_drop", [0.0, 0.2]),
    ],
    "adaboost": [("max_depth", [4, 6, 8])],
    "cnn": [("learning_rate", [0.01, 0.03, 0.1]), ("channels", [(8, 16), (16, 32)])],
}
LIGHT_CURVE = {"adaboost": 200, "softmax": 60, "svm": 60}


def _jsonable(v):
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def _key(params):
    return json.dumps({k: _jsonable(v) for k, v in sorted(params.items())}, sort_keys=True)


def _round_curve(values, nd=5):
    return [round(float(v), nd) for v in values]


def run_trial(model_key, params, data):
    """Fit one configuration on train, evaluate on train and *validation*. Returns a record."""
    spec = SPECS[model_key]
    model = spec["cls"](**params)
    t0 = time.perf_counter()
    model.fit(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    ) if model_key != "naive_bayes" else model.fit(data["X_train"], data["y_train"])
    train_time = time.perf_counter() - t0
    out = {"params": {k: _jsonable(v) for k, v in params.items()}, "train_time_sec": train_time}
    for split in ("train", "val"):
        m = get_metrics(data[f"y_{split}"], model.predict(data[f"X_{split}"]), labels=range(10))
        out[f"{split}_acc"] = m["accuracy"]
        out[f"{split}_f1_macro"] = m["f1_macro"]
    h = getattr(model, "history", {}) or {}
    out["curve"] = {k: _round_curve(v) for k, v in h.items() if len(v) and k != "lr"}
    if hasattr(model, "stopped_epoch_"):
        out["stopped_epoch"] = int(model.stopped_epoch_)
        out["best_epoch"] = int(model.best_epoch_)
    return out


def subset(data, n_train=None, n_val=None):
    """Deterministic class-balanced-ish prefix of the (already shuffled) split."""
    d = dict(data)
    if n_train:
        d["X_train"], d["y_train"] = data["X_train"][:n_train], data["y_train"][:n_train]
    if n_val:
        d["X_val"], d["y_val"] = data["X_val"][:n_val], data["y_val"][:n_val]
    return d


def tune(model_key, data, tracker=None, smoke=False, out_dir=None, light=False, log=print):
    """Run the sequential sweeps. ``data`` must contain train and validation arrays only."""
    spec = SPECS[model_key]
    base = dict(spec["base"])
    sweeps = list(spec["sweeps"])
    if light:
        sweeps = list(LIGHT[model_key])
    if smoke:
        base.update(SMOKE["base_override"].get(model_key, {}))
        sweeps = SMOKE["sweeps"][model_key]
        data = subset(data, SMOKE["n_train"], SMOKE["n_val"])
    cname = CLASS_NAME[model_key]
    trials, cache = [], {}
    best = dict(base)

    def evaluate(params, stage, param, value):
        k = _key(params)
        if k in cache:
            return cache[k]
        run_name = f"{cname}:{param}={value}" if param else f"{cname}:base"
        cfg = {
            "model": cname,
            "stage": stage,
            "seed": SEED,
            "smoke": smoke,
            **{f"hp/{a}": _jsonable(b) for a, b in params.items()},
        }
        if tracker is not None:
            with tracker.run(
                run_name, group=f"tune-{cname}", job_type="tune", config=cfg, tags=["tune", cname]
            ) as run:
                rec = run_trial(model_key, params, data)
                _log_trial(run, rec)
        else:
            rec = run_trial(model_key, params, data)
        rec.update({"stage": stage, "param": param, "value": _jsonable(value)})
        cache[k] = rec
        trials.append(rec)
        log(
            f"[{cname}] {stage:<24} {str(value):<14} val_acc={rec['val_acc']:.4f} "
            f"train_acc={rec['train_acc']:.4f} time={rec['train_time_sec']:.1f}s"
        )
        return rec

    for param, values in sweeps:
        stage = f"sweep:{param}"
        scored = []
        for v in values:
            p = {**best, param: v}
            scored.append((evaluate(p, stage, param, v)["val_acc"], v))
        top = max(s for s, _ in scored)
        best[param] = next(v for s, v in scored if s == top)  # first maximum -> simplest value
        log(f"[{cname}] -> {param} = {best[param]}")

    curve_info = None
    if "epoch_curve" in spec:
        ec = spec["epoch_curve"]
        cap = (
            SMOKE["curve_max"][model_key]
            if smoke
            else LIGHT_CURVE.get(model_key, ec["max"])
            if light
            else ec["max"]
        )
        params = {**best, ec["param"]: cap}
        rec = evaluate(params, f"curve:{ec['param']}", ec["param"], cap)
        accs = rec["curve"]["val_acc"]
        n_best = int(np.argmax(accs)) + 1
        best[ec["param"]] = n_best
        curve_info = {
            "param": ec["param"],
            "max": cap,
            "chosen": n_best,
            "val_acc_at_chosen": accs[n_best - 1],
        }
        log(f"[{cname}] -> {ec['param']} = {n_best} (val_acc {accs[n_best - 1]:.4f})")

    best_val = curve_info["val_acc_at_chosen"] if curve_info else max(t["val_acc"] for t in trials)
    result = {
        "model": cname,
        "protocol": "sequential one-factor sweeps; selection by validation accuracy only",
        "smoke": smoke,
        "light": light,
        "n_train": int(len(data["y_train"])),
        "n_val": int(len(data["y_val"])),
        "seed": SEED,
        "git": git_info(),
        "hardware": hardware_info(),
        "n_trials": len(trials),
        "best_params": {k: _jsonable(v) for k, v in best.items()},
        "best_val_acc": best_val,
        "epoch_curve": curve_info,
        "trials": trials,
    }
    out_dir = out_dir or RESULTS_DIR / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cname}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _log_trial(run, rec):
    curve = rec.get("curve", {})
    n = max((len(v) for v in curve.values()), default=0)
    for i in range(n):
        row = {}
        for src, dst in (
            ("loss", "train/loss"),
            ("inner_val_loss", "inner_val/loss"),
            ("val_loss", "val/loss"),
            ("val_acc", "val/acc"),
            ("train_acc", "train/acc"),
        ):
            if src in curve and i < len(curve[src]):
                row[dst] = curve[src][i]
        run.log(row, step=i + 1)
    run.summary(
        {
            "train/acc": rec["train_acc"],
            "val/acc": rec["val_acc"],
            "val/f1_macro": rec["val_f1_macro"],
            "train_time_sec": rec["train_time_sec"],
        }
    )
