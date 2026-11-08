"""Plain vs augmented variants of every model: clean accuracy and robustness on the test split.

For each model kind the plain model (``artifacts/<Class>``) and the augmented variant
(``artifacts/<Class>_aug``, trained on the original images plus randomly rotated / shifted / scaled
copies) are evaluated on the same test images: clean, rotated by a random angle in +-a degrees
(``ROTATION_ANGLES``), and under each fixed perturbation of ``perturb_benchmark``. The test split is
used for reporting only; nothing here selects anything.
"""

import json

import numpy as np

from src.experiments.augment import ROTATION_ANGLES, perturb_benchmark, transform_images
from src.experiments.search import CLASS_NAME
from src.models.base import BaseClassifier
from src.utils.paths import ARTIFACTS_DIR, RESULTS_DIR

KINDS = ["naive_bayes", "softmax", "svm", "mlp", "adaboost", "cnn"]


def _acc(model, X, y):
    return float(np.mean(model.predict(X) == y))


def evaluate_variant(model, data, seed=999):
    X, y = data["X_test"], data["y_test"]
    rot = {
        str(a): _acc(
            model,
            X if a == 0 else transform_images(X, ("rotation",), k=1, seed=seed, rotation=float(a)),
            y,
        )
        for a in ROTATION_ANGLES
    }
    stress = {t: _acc(model, Xs, y) for t, Xs in perturb_benchmark(X, seed=seed).items()}
    return {"test_rotation_acc": rot, "test_stress_acc": stress}


def run_pairs(data, art_root=None, res_root=None, kinds=KINDS, log=print):
    art_root = art_root or ARTIFACTS_DIR
    res_root = res_root or RESULTS_DIR
    out = {
        "schema": 1,
        "rotation_angles": list(ROTATION_ANGLES),
        "note": "rotation a: every test image rotated by a random angle in [-a, a] degrees",
        "models": {},
    }
    for kind in kinds:
        cname = CLASS_NAME[kind]
        row = {}
        for tag, name in (("plain", cname), ("augmented", cname + "_aug")):
            npz = art_root / name / "model.npz"
            fin = res_root / "final" / f"{name}.json"
            if not npz.is_file() or not fin.is_file():
                raise SystemExit(f"{npz} / {fin} not found - train {kind} (with --augment) first.")
            f = json.loads(fin.read_text(encoding="utf-8"))
            s = f["splits"]
            row[tag] = {
                "class": name,
                "train_acc": s["train"]["accuracy"],
                "val_acc": s["val"]["accuracy"],
                "test_acc": s["test"]["accuracy"],
                "test_f1_macro": s["test"]["f1_macro"],
                "fit_time_sec": f["train_time_sec"],
                "latency_ms": f["latency_ms_single_sample"],
                "augmentation": f.get("augmentation"),
                **evaluate_variant(BaseClassifier.load(npz), data),
            }
            log(
                f"[pairs] {kind:12s} {tag:9s} test={row[tag]['test_acc']:.4f} "
                f"rot30={row[tag]['test_rotation_acc']['30']:.4f} rot90={row[tag]['test_rotation_acc']['90']:.4f}"
            )
        out["models"][kind] = row
    path = res_root / "augmentation_pairs.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
