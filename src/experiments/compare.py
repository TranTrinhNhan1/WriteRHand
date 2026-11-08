"""CNN vs MLP comparison and the (explicitly labelled) cross-dataset evaluation.

Everything is measured on models that were trained with the project protocol:

* ``cnn_vs_mlp``: parameters, ONNX size, accuracy / macro metrics (from ``results/final``), training
  and inference times (same machine, same runs), generalisation gap, an exact McNemar test between
  their test predictions, and accuracy under fixed stress perturbations of the test images
  (rotation, translation, scaling, Gaussian noise, salt-and-pepper; ``augment.STRESS``). The
  perturbations are fixed (seed 999), identical for both models, and used only for reporting.
* ``cross_eval``: a model trained on dataset A evaluated on the test set of dataset B. Only done
  between datasets with the same class definition (MNIST and EMNIST Digits both label the digits
  0-9); it is NOT the model trained on B and is shown as such. Fashion-MNIST has different classes,
  so no cross evaluation is run for it.
"""

import json
from pathlib import Path

import numpy as np

from src.experiments.augment import TRANSFORMS, perturb_benchmark
from src.models.base import BaseClassifier
from src.preprocessing.preprocess import DATASETS, load_dataset, load_raw


def _final(rd, cname):
    p = Path(rd) / "final" / f"{cname}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def mcnemar_exact(pred_a, pred_b, y):
    from scipy.stats import binomtest

    a, b = pred_a == y, pred_b == y
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    return only_a, only_b, float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0


def cnn_vs_mlp(dataset, results_dir, artifacts_dir, onnx_bytes=None):
    data, _ = load_dataset(name=dataset)
    X, y = data["X_test"], data["y_test"]
    stress = perturb_benchmark(X)
    rows, preds = [], {}
    for mid, cname in (("mlp", "MLP"), ("cnn", "CNN")):
        fin = _final(results_dir, cname)
        npz = Path(artifacts_dir) / cname / "model.npz"
        if fin is None or not npz.is_file():
            return None
        model = BaseClassifier.load(npz)
        preds[mid] = model.predict(X)
        s = fin["splits"]
        n_params = (
            model.n_parameters()
            if hasattr(model, "n_parameters")
            else int(sum(w.numel() for w in model.W) + sum(b.numel() for b in model.b))
        )
        rows.append(
            {
                "id": mid,
                "n_parameters": n_params,
                "onnx_bytes": (onnx_bytes or {}).get(mid),
                "train_time_sec": fin["train_time_sec"],
                "latency_ms": fin["latency_ms_single_sample"],
                "predict_time_10k_sec": s["test"]["predict_time_sec"]
                * 10_000
                / s["test"]["n_samples"],
                "epochs": fin.get("stopped_epoch"),
                "accuracy": {k: s[k]["accuracy"] for k in ("train", "val", "test")},
                "precision_macro": s["test"]["precision_macro"],
                "recall_macro": s["test"]["recall_macro"],
                "f1_macro": s["test"]["f1_macro"],
                "generalization_gap": s["train"]["accuracy"] - s["test"]["accuracy"],
                "robust": {
                    "clean": s["test"]["accuracy"],
                    **{t: float(np.mean(model.predict(stress[t]) == y)) for t in TRANSFORMS},
                },
            }
        )
    only_mlp, only_cnn, p = mcnemar_exact(preds["mlp"], preds["cnn"], y)
    return {
        "dataset": dataset,
        "rows": rows,
        "mcnemar": {"only_mlp": only_mlp, "only_cnn": only_cnn, "p": p},
        "stress": "fixed perturbations of the test images (augment.STRESS, seed 999); reporting only",
        "interpretability_note": (
            "Both models are inspected with the same tool (execution explorer). MLP first-layer "
            "neurons have one weight per pixel and can be drawn as 28x28 maps; CNN kernels are small "
            "(k x k) and shared across positions, so their weights are easy to draw but a single "
            "kernel says little about the final decision without the following stages. This is a "
            "qualitative statement, not a measured score."
        ),
    }


def cross_eval(src_dataset, dst_dataset, src_artifacts, models=None):
    """Accuracy of models trained on ``src_dataset`` on the test set of ``dst_dataset``."""
    if DATASETS[src_dataset]["classes"] != DATASETS[dst_dataset]["classes"]:
        raise ValueError("cross evaluation needs identical class definitions")
    _, meta = load_dataset(name=src_dataset)
    _, _, Xte, yte = load_raw(dst_dataset)
    X = ((Xte.reshape(-1, 784).float() / 255.0).numpy().astype(np.float64) - meta["mean"]) / (
        meta["std"] + 1e-8
    )
    y = yte.numpy()
    rows = []
    for mid, cname in models or (
        ("naive_bayes", "GaussianNB"),
        ("softmax", "SoftmaxRegression"),
        ("mlp", "MLP"),
        ("cnn", "CNN"),
        ("adaboost", "AdaBoostSAMME"),
    ):
        npz = Path(src_artifacts) / cname / "model.npz"
        if npz.is_file():
            rows.append(
                {"model": mid, "accuracy": float(np.mean(BaseClassifier.load(npz).predict(X) == y))}
            )
    return {
        "trained_on": src_dataset,
        "evaluated_on": dst_dataset,
        "rows": rows,
        "note": (
            f"Models trained on {DATASETS[src_dataset]['title']} (its own normalisation constants) "
            f"evaluated on the {DATASETS[dst_dataset]['title']} test set. These are NOT the models "
            "trained on that dataset; see the dataset's own table for those."
        ),
    }
