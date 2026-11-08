"""Stacking experiment: meta-learner over the six trained base models (hold-out = validation split).

Protocol (no test leakage):
1. load the base models trained on the 50k train split (``artifacts/<Model>/model.npz``);
2. level-1 features of the validation split; the meta-learner's L2 strength is chosen by 5-fold
   cross-validation *inside* the validation split (that CV accuracy is the honest validation estimate);
3. fit the meta-learner on the whole validation split, then report train / validation / test.
   The validation accuracy of the final meta-learner is in-sample for the meta-learner and is labelled so;
4. the test split is used once, for reporting.
A soft-voting reference (mean of the probabilistic base models' probabilities, nothing learned) is reported too.
"""

import json
import time

import numpy as np

from src.evaluation.evaluator import Evaluator
from src.experiments.env import git_info, hardware_info
from src.experiments.final import single_sample_latency_ms
from src.models.base import BaseClassifier
from src.models.stacking import StackingEnsemble
from src.utils.paths import ARTIFACTS_DIR, RESULTS_DIR

BASE_NAMES = ["GaussianNB", "SoftmaxRegression", "MLP", "AdaBoostSAMME", "CNN", "LinearSVM"]
L2_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
FOLDS = 5
NAME = "StackingEnsemble"


def load_bases(art_root, names=BASE_NAMES):
    bases = []
    for n in names:
        path = art_root / n / "model.npz"
        if not path.is_file():
            raise SystemExit(
                f"{path} not found - train every base model first (`train --model all`)."
            )
        bases.append((n, BaseClassifier.load(path)))
    return bases


def cv_accuracy(F, y, l2, folds=FOLDS, seed=42):
    """Mean / sd of the fold accuracies of a meta-learner with the given L2 (features already computed)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    parts = np.array_split(order, folds)
    accs = []
    for i in range(folds):
        va = parts[i]
        tr = np.concatenate([p for j, p in enumerate(parts) if j != i])
        m = StackingEnsemble(l2=l2).fit_features(F[tr], y[tr])
        accs.append(float(np.mean(m.predict_from_features(F[va]) == y[va])))
    return float(np.mean(accs)), float(np.std(accs, ddof=1))


def run_stacking(data, meta, art_root=None, out_dir=None, art_dir=None, repeats=5, log=print):
    art_root = art_root or ARTIFACTS_DIR
    out_dir = out_dir or RESULTS_DIR / "final"
    art_dir = art_dir or art_root / NAME
    bases = load_bases(art_root)
    stack = StackingEnsemble(bases=bases)

    t0 = time.perf_counter()
    F_val = stack.features(data["X_val"])
    feat_time = time.perf_counter() - t0

    cv = {}
    for l2 in L2_GRID:
        mean, sd = cv_accuracy(F_val, data["y_val"], l2)
        cv[l2] = {"cv_val_acc_mean": mean, "cv_val_acc_sd": sd}
        log(f"[{NAME}] l2={l2:<7} 5-fold CV on validation: {mean:.4f} +- {sd:.4f}")
    best_l2 = max(L2_GRID, key=lambda v: cv[v]["cv_val_acc_mean"])
    stack.l2 = best_l2

    t0 = time.perf_counter()
    stack.fit_features(F_val, data["y_val"])
    meta_fit = time.perf_counter() - t0

    ev = Evaluator(NAME, stack)
    ev.results["train_time_sec"] = meta_fit
    for split, r in (("train", 1), ("val", repeats), ("test", repeats)):
        ev.evaluate(data[f"X_{split}"], data[f"y_{split}"], split, repeats=r)

    # reference: soft voting (nothing learned) and every base model on the same test images
    probs_test = {n: m.predict_proba(data["X_test"]) for n, m in bases if n != "LinearSVM"}
    vote = np.mean(list(probs_test.values()), axis=0).argmax(axis=1)
    classes = stack.classes
    soft_vote_test = float(np.mean(classes[vote] == data["y_test"]))
    base_test = {n: float(np.mean(m.predict(data["X_test"]) == data["y_test"])) for n, m in bases}

    res = ev.results
    res.update(
        {
            "params": {k: v for k, v in stack.get_params().items()},
            "seed": stack.seed,
            "base_models": [n for n, _ in bases],
            "meta_learner": "SoftmaxRegression (own implementation) on standardised log-probabilities / SVM scores",
            "n_features": int(F_val.shape[1]),
            "meta_fit_sec": meta_fit,
            "base_features_sec_on_validation": feat_time,
            "cv_on_validation": {str(k): v for k, v in cv.items()},
            "selected_l2": best_l2,
            "cv_val_acc": cv[best_l2]["cv_val_acc_mean"],
            "soft_vote_test_accuracy": soft_vote_test,
            "base_test_accuracy": base_test,
            "block_importance": stack.block_importance(),
            "latency_ms_single_sample": single_sample_latency_ms(stack, data["X_test"]),
            "dataset": meta,
            "git": git_info(),
            "hardware": hardware_info(),
            "protocol": "meta-learner fitted on the validation split (hold-out of the base models); "
            "L2 chosen by 5-fold CV inside the validation split; test evaluated once. The train and "
            "validation accuracies are not clean estimates for a stacked model: the base models were fitted "
            "on train, and the meta-learner on validation.",
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ev.save(out_dir)
    np.savez_compressed(
        out_dir / f"{NAME}_predictions.npz",
        **{k: v.astype(np.int8) for k, v in ev.predictions.items()},
    )
    art_dir.mkdir(parents=True, exist_ok=True)
    stack.save(art_dir / "model.npz")
    log(
        f"[{NAME}] train={res['splits']['train']['accuracy']:.4f} val(in-sample)="
        f"{res['splits']['val']['accuracy']:.4f} cv_val={res['cv_val_acc']:.4f} "
        f"test={res['splits']['test']['accuracy']:.4f} soft_vote_test={soft_vote_test:.4f}"
    )
    return stack, res


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
