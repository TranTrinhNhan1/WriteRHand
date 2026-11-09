"""Stratified K-fold evaluation of the six models with the hyper-parameters already chosen.

Protocol
* The pool is the whole official training set (MNIST: 60 000 images = the 50 000 / 10 000 train / validation
  split put back together). It is cut into ``k`` stratified, seeded folds; fold ``i`` is held out, the other
  ``k - 1`` folds are the training part. The official test set is the same for every fold and is only
  reported, never used to decide anything.
* The hyper-parameters are the ones chosen by ``tune`` on the original validation split (``results/tuning``);
  they are **not** re-tuned inside the folds. The original validation images are part of the pool, so the
  fold scores carry a small optimism from that choice; this is stated in every result file.
* Standardisation uses the mean / std of the fold's training part only (nothing leaks from the held-out fold
  or from the test set).
* Early stopping (MLP, CNN) uses an internal split of the training part, never the held-out fold.

``results/kfold/<Model>.json`` is written after every fold, so an interrupted run resumes where it stopped.
"""

import json
import time

import numpy as np

from src.evaluation.classification_metrics import get_metrics
from src.experiments.env import git_info, hardware_info
from src.experiments.search import CLASS_NAME, SPECS
from src.preprocessing.preprocess import flatten, load_raw
from src.utils.paths import RESULTS_DIR

SEED = 42
K = 5


def stratified_folds(y, k=K, seed=SEED):
    """``k`` sorted index arrays that partition ``range(len(y))``; each keeps the class proportions."""
    y = np.asarray(y).astype(np.int64)
    if not 2 <= k <= len(y):
        raise ValueError("k must satisfy 2 <= k <= len(y)")
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(k)]
    for n, c in enumerate(np.unique(y)):
        idx = rng.permutation(np.flatnonzero(y == c))
        # array_split makes the first chunks one image larger; rotating them per class keeps the fold
        # sizes within a few images of each other
        for i, chunk in enumerate(np.array_split(idx, k)):
            parts[(i + n) % k].append(chunk)
    return [np.sort(np.concatenate(p)) for p in parts]


def fold_arrays(X_raw, y, X_test_raw, folds, i):
    """Standardised ``(X_train, y_train, X_val, y_val, X_test)`` of fold ``i`` (statistics from its training part)."""
    val_idx = folds[i]
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    tr = np.flatnonzero(mask)
    mean = float(X_raw[tr].mean(dtype=np.float64))
    std = float(X_raw[tr].std(dtype=np.float64, ddof=1))

    def norm(a):
        return ((a - mean) / (std + 1e-8)).astype(np.float32)

    return norm(X_raw[tr]), y[tr], norm(X_raw[val_idx]), y[val_idx], norm(X_test_raw), mean, std


def _params(model_key, results_dir):
    path = results_dir / "tuning" / f"{CLASS_NAME[model_key]}.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found - run `tune --model {model_key}` first.")
    return json.loads(path.read_text(encoding="utf-8"))["best_params"]


def _build(model_key, params):
    cls = SPECS[model_key]["cls"]
    return cls(
        **{
            k: tuple(v) if k in ("hidden_sizes", "channels", "dense") else v
            for k, v in params.items()
        }
    )


def _merge_write(path, res, fold):
    """Add one finished fold to the result file, re-reading it first so that processes that train
    different folds of the same model at the same time do not overwrite each other."""
    cur = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else res
    cur["folds"][fold] = res["folds"][fold]
    for key in ("git", "hardware", "note"):
        cur[key] = res[key]
    tmp = path.with_suffix(f".{fold}.tmp")
    tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_model(
    model_key, k=K, seed=SEED, only_folds=None, out_dir=None, results_dir=RESULTS_DIR, log=print
):
    """Train and evaluate ``model_key`` on the requested folds (default: all) and update its result file."""
    name = CLASS_NAME[model_key]
    out_dir = out_dir or results_dir / "kfold"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    params = _params(model_key, results_dir)

    Xtr_raw, ytr_raw, Xte_raw, yte_raw = load_raw("mnist")
    X_raw = flatten(Xtr_raw).numpy()
    y = ytr_raw.numpy().astype(np.int64)
    X_test_raw = flatten(Xte_raw).numpy()
    y_test = yte_raw.numpy().astype(np.int64)
    folds = stratified_folds(y, k, seed)

    res = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {
            "model": name,
            "k": k,
            "seed": seed,
            "params": params,
            "n_pool": int(len(y)),
            "n_test": int(len(y_test)),
            "folds": {},
        }
    )
    if res["k"] != k or res["seed"] != seed or res["params"] != params:
        raise SystemExit(f"{path} was produced with other k / seed / params; remove it to restart.")

    for i in range(k) if only_folds is None else only_folds:
        if str(i) in res["folds"]:
            log(f"[{name}] fold {i + 1}/{k} already done, skipped")
            continue
        Xtr, ytr, Xva, yva, Xte, mean, std = fold_arrays(X_raw, y, X_test_raw, folds, i)
        model = _build(model_key, params)
        kw = {} if model_key == "naive_bayes" else {"X_val": Xva, "y_val": yva}
        t0 = time.perf_counter()
        model.fit(Xtr, ytr, **kw)
        fit_s = time.perf_counter() - t0
        m_val = get_metrics(yva, model.predict(Xva), labels=range(10))
        m_test = get_metrics(y_test, model.predict(Xte), labels=range(10))
        keep = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        res["folds"][str(i)] = {
            "n_train": int(len(ytr)),
            "n_val": int(len(yva)),
            "mean": mean,
            "std": std,
            "val": {m: float(m_val[m]) for m in keep},
            "test": {m: float(m_test[m]) for m in keep},
            "train_time_sec": round(fit_s, 2),
            **(
                {"stopped_epoch": int(model.stopped_epoch_)}
                if hasattr(model, "stopped_epoch_")
                else {}
            ),
        }
        res["git"] = git_info()
        res["hardware"] = hardware_info()
        res["note"] = (
            "hyper-parameters were chosen on the original validation split (a subset of this pool) and are "
            "not re-tuned per fold; standardisation uses the fold's training part; test is never used for any choice"
        )
        _merge_write(path, res, str(i))
        log(
            f"[{name}] fold {i + 1}/{k}: held-out acc={m_val['accuracy']:.4f} "
            f"test acc={m_test['accuracy']:.4f} fit={fit_s:.1f}s"
        )
    return res


def summarise(out_dir=None, results_dir=RESULTS_DIR):
    """Mean and standard deviation (ddof=1) over the finished folds of every model; writes ``summary.json``."""
    out_dir = out_dir or results_dir / "kfold"
    rows = {}
    for name in CLASS_NAME.values():
        path = out_dir / f"{name}.json"
        if not path.is_file():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        done = [r["folds"][f] for f in sorted(r["folds"], key=int)]
        if not done:
            continue

        def agg(split, metric, done=done):
            v = np.array([f[split][metric] for f in done])
            return {
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else None,
                "values": [float(x) for x in v],
            }

        rows[name] = {
            "k": r["k"],
            "n_folds_done": len(done),
            "val_accuracy": agg("val", "accuracy"),
            "val_f1_macro": agg("val", "f1_macro"),
            "test_accuracy": agg("test", "accuracy"),
            "test_f1_macro": agg("test", "f1_macro"),
            "train_time_sec_mean": float(np.mean([f["train_time_sec"] for f in done])),
        }
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
