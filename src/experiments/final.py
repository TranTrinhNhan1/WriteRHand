"""Final training / evaluation, scikit-learn reference models and artifact saving."""

import json
import time

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.evaluator import Evaluator
from src.experiments.env import git_info, hardware_info
from src.experiments.search import CLASS_NAME, SPECS
from src.utils.paths import ARTIFACTS_DIR, RESULTS_DIR


def single_sample_latency_ms(model, X, n=200, seed=0):
    """Median latency of ``predict`` on ONE row (what an interactive demo experiences)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), size=n)
    model.predict(X[:1])  # warm-up
    times = []
    for i in idx:
        t0 = time.perf_counter()
        model.predict(X[i : i + 1])
        times.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(times))


def sklearn_counterpart(model_key, params, n_train):
    """scikit-learn model configured as closely as possible to ours (no extra tuning)."""
    p = dict(params)
    if model_key == "naive_bayes":
        return SkGaussianNB(var_smoothing=p["var_smoothing"])
    if model_key == "softmax":
        # sklearn: C * sum CE + 0.5 ||W||^2   <=>   mean CE + 1/(2 C n) ||W||^2  =>  C = 1/(l2 n)
        kw = {"penalty": None} if p["l2"] == 0 else {"C": 1.0 / (p["l2"] * n_train)}
        return SkLogisticRegression(solver="lbfgs", max_iter=1000, tol=1e-6, **kw)
    if model_key == "svm":
        # sklearn: 0.5 ||w||^2 + C sum_i phi_i   <=>   mean phi + l2/2 ||w||^2   with   C = 1 / (l2 n)
        # (liblinear also penalises the intercept; ours does not)
        c = 1e3 if p["l2"] == 0 else 1.0 / (p["l2"] * n_train)
        hinge = p["loss"] == "hinge"
        return LinearSVC(
            loss=p["loss"], C=c, dual=hinge, tol=1e-4, max_iter=2000, random_state=p["seed"]
        )
    if model_key == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=tuple(p["hidden_sizes"]),
            activation="relu",
            solver="sgd",
            alpha=0.0,
            batch_size=p["batch_size"],
            learning_rate_init=p["learning_rate"],
            momentum=p["momentum"],
            nesterovs_momentum=False,
            learning_rate="constant",
            early_stopping=True,
            validation_fraction=0.08,
            n_iter_no_change=5,
            tol=1e-4,
            max_iter=p["epochs"],
            random_state=p["seed"],
        )
    if model_key == "adaboost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=p["max_depth"], random_state=p["seed"]),
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            random_state=p["seed"],
        )
    raise KeyError(model_key)


def _evaluate_all(name, model, data, fit_kwargs, include_test, repeats, fit_xy=None):
    """Fit on ``fit_xy`` (default: the train split); train accuracy is always on the original train images."""
    ev = Evaluator(name, model)
    Xf, yf = fit_xy if fit_xy is not None else (data["X_train"], data["y_train"])
    ev.fit(Xf, yf, **fit_kwargs)
    ev.evaluate(data["X_train"], data["y_train"], "train", repeats=1)
    ev.evaluate(data["X_val"], data["y_val"], "val", repeats=repeats)
    if include_test:
        ev.evaluate(data["X_test"], data["y_test"], "test", repeats=repeats)
    return ev


def train_final(
    model_key,
    params,
    data,
    meta,
    tracker=None,
    save=True,
    repeats=5,
    out_dir=None,
    art_dir=None,
    log=print,
    augment=None,
):
    """Train the selected configuration on the 50k train split and report train / val / test.

    ``augment`` (a spec of ``augment.deploy_spec``) adds randomly transformed copies to the images the model
    is fitted on; train / validation / test accuracies are still measured on the original images.
    """
    cls = SPECS[model_key]["cls"]
    cname = CLASS_NAME[model_key] + ("_aug" if augment else "")
    out_dir = out_dir or RESULTS_DIR / "final"
    art_dir = art_dir or ARTIFACTS_DIR / cname
    model = cls(
        **{
            k: tuple(v) if k in ("hidden_sizes", "channels", "dense") else v
            for k, v in params.items()
        }
    )
    fit_kwargs = (
        {} if model_key == "naive_bayes" else {"X_val": data["X_val"], "y_val": data["y_val"]}
    )
    fit_xy = None
    if augment:
        from src.experiments.augment import apply_spec

        fit_xy = apply_spec(data["X_train"], data["y_train"], augment)
    ev = _evaluate_all(
        cname, model, data, fit_kwargs, include_test=True, repeats=repeats, fit_xy=fit_xy
    )
    latency = single_sample_latency_ms(model, data["X_test"])
    res = ev.results
    res.update(
        {
            "params": params,
            "seed": params.get("seed"),
            "latency_ms_single_sample": latency,
            "dataset": meta,
            "git": git_info(),
            "hardware": hardware_info(),
            "history": {
                k: [round(float(x), 5) for x in v]
                for k, v in (model.history or {}).items()
                if len(v) and k != "lr"
            },
            "protocol": "trained on the 50k train split with the configuration chosen on validation; "
            "test evaluated once for this configuration"
            + (
                "; the training images are extended with randomly transformed copies (see `augmentation`)"
                if augment
                else ""
            ),
            "augmentation": augment,
        }
    )
    if hasattr(model, "stopped_epoch_"):
        res.update(
            {"stopped_epoch": int(model.stopped_epoch_), "best_epoch": int(model.best_epoch_)}
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    ev.save(out_dir)
    np.savez_compressed(
        out_dir / f"{cname}_predictions.npz",
        **{k: v.astype(np.int8) for k, v in ev.predictions.items()},
    )
    if save:
        art_dir.mkdir(parents=True, exist_ok=True)
        model.save(art_dir / "model.npz")
    if tracker is not None:
        _log_final(tracker, cname, res, ev, save and art_dir / "model.npz")
    log(
        f"[{cname}] train={res['splits']['train']['accuracy']:.4f} "
        f"val={res['splits']['val']['accuracy']:.4f} test={res['splits']['test']['accuracy']:.4f} "
        f"fit={res['train_time_sec']:.1f}s"
    )
    return model, res


def run_baseline(model_key, params, data, repeats=5, log=print):
    """scikit-learn counterpart trained / evaluated under the same protocol (test included)."""
    skm = sklearn_counterpart(model_key, params, len(data["y_train"]))
    ev = _evaluate_all(f"sklearn_{CLASS_NAME[model_key]}", skm, data, {}, True, repeats)
    res = ev.results
    res.update(
        {
            "sklearn_class": type(skm).__name__,
            "sklearn_params": {
                k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in skm.get_params().items()
            },
            "latency_ms_single_sample": single_sample_latency_ms(skm, data["X_test"]),
            "note": "configured to mirror our model; not separately tuned",
            "predictions": {k: v.astype(np.int8) for k, v in ev.predictions.items()},
        }
    )
    log(
        f"[sklearn {CLASS_NAME[model_key]}] test={res['splits']['test']['accuracy']:.4f} "
        f"fit={res['train_time_sec']:.1f}s"
    )
    return res


def _log_final(tracker, cname, res, ev, model_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = {
        "model": cname,
        "seed": res["seed"],
        "git_commit": res["git"]["commit"],
        "split_fingerprint": res["dataset"]["split_fingerprint"],
        "preprocess_version": res["dataset"]["preprocess_version"],
        **{f"hp/{k}": v for k, v in res["params"].items()},
    }
    with tracker.run(
        f"{cname}:final", group="final", job_type="final", config=cfg, tags=["final", cname]
    ) as run:
        hist = res.get("history", {})
        n = max((len(v) for v in hist.values()), default=0)
        for i in range(n):
            row = {f"curve/{k}": v[i] for k, v in hist.items() if i < len(v)}
            run.log(row, step=i + 1)
        summary = {
            "train_time_sec": res["train_time_sec"],
            "latency_ms_single_sample": res["latency_ms_single_sample"],
        }
        for split, m in res["splits"].items():
            summary.update(
                {
                    f"{split}/accuracy": m["accuracy"],
                    f"{split}/precision_macro": m["precision_macro"],
                    f"{split}/recall_macro": m["recall_macro"],
                    f"{split}/f1_macro": m["f1_macro"],
                    f"{split}/predict_time_sec": m["predict_time_sec"],
                }
            )
        run.summary(summary)
        for split in ("val", "test"):
            cm = np.array(res["splits"][split]["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(5, 4.5))
            ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{cname} - {split}")
            run.log_figure(f"confusion_matrix/{split}", fig)
            plt.close(fig)
        if model_path:
            run.log_artifact(
                model_path,
                f"{cname}-model",
                "model",
                {"test_accuracy": res["splits"]["test"]["accuracy"]},
            )


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
