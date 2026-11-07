"""Assemble the deployable bundle (``artifacts/bundle``) that is uploaded to Hugging Face.

bundle/
  manifest.json          version, preprocessing, class labels, per-model files + test metrics
  models/<id>.onnx       mlp, softmax, naive_bayes  (input: pixels/255, output: probabilities)
  models/adaboost.json   SAMME ensemble (evaluated by a small JS function in the browser)
  python/<Class>.npz     original NumPy/torch weights (load with BaseClassifier.load)
  test_vectors.json      raw test digits + expected outputs (browser/ONNX parity tests)
  results.json           metrics used by the website dashboard
  README.md              model card
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.export.onnx_export import (
    adaboost_predict_json,
    adaboost_to_json,
    gaussian_nb_to_onnx,
    mlp_to_onnx,
    softmax_regression_to_onnx,
)
from src.models.base import BaseClassifier
from src.preprocessing.preprocess import PREPROCESS_VERSION, load_dataset, load_mnist

MODEL_VERSION = "1.0.0"

# id used in the website -> (class name, ONNX builder or None for JSON)
EXPORTS = {
    "naive_bayes": ("GaussianNB", gaussian_nb_to_onnx),
    "softmax": ("SoftmaxRegression", softmax_regression_to_onnx),
    "mlp": ("MLP", mlp_to_onnx),
    "adaboost": ("AdaBoostSAMME", None),
}
TITLES = {
    "naive_bayes": "Gaussian Naive Bayes",
    "softmax": "Softmax regression",
    "mlp": "Multilayer perceptron",
    "adaboost": "AdaBoost (SAMME)",
}


def _validate_onnx(proto, P, ref_proba, tol):
    import onnxruntime as ort

    sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    out = sess.run(["probabilities"], {"pixels": P})[0]
    diff = float(np.abs(out - ref_proba).max())
    agree = float((out.argmax(1) == ref_proba.argmax(1)).mean())
    return {
        "max_abs_prob_diff": diff,
        "argmax_agreement": agree,
        "n_checked": int(len(P)),
        "ok": diff <= tol and agree >= 0.999,
    }


def export_all(artifacts_dir, results_dir, n_vectors=24, log=print):
    import onnx

    bundle = Path(artifacts_dir) / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "models").mkdir(parents=True)
    (bundle / "python").mkdir()

    _, meta = load_dataset()
    mean, std = meta["mean"], meta["std"]
    _, _, Xte_raw, yte = load_mnist()
    P = (Xte_raw.reshape(-1, 784).float() / 255.0).numpy()  # raw test pixels in [0,1]
    y = yte.numpy()
    Xn = (P.astype(np.float64) - mean) / (std + 1e-8)  # what the Python models were trained on
    rng = np.random.default_rng(0)
    vec_idx = np.sort(rng.choice(len(P), n_vectors, replace=False))

    manifest_models, vectors = (
        {},
        {
            "indices": vec_idx.tolist(),
            "labels": y[vec_idx].tolist(),
            "pixels": (P[vec_idx] * 255).round().astype(int).tolist(),
            "expected": {},
        },
    )
    for mid, (cname, builder) in EXPORTS.items():
        npz = Path(artifacts_dir) / cname / "model.npz"
        if not npz.is_file():
            log(f"skip {mid}: {npz} not found")
            continue
        model = BaseClassifier.load(npz)
        shutil.copy(npz, bundle / "python" / f"{cname}.npz")
        ref = model.predict_proba(Xn)
        if builder is not None:
            proto = builder(model, mean, std)
            path = bundle / "models" / f"{mid}.onnx"
            onnx.save(proto, path)
            tol = 1e-3 if mid == "naive_bayes" else 1e-4
            check = _validate_onnx(proto, P.astype(np.float32), ref, tol)
            fmt, fname = "onnx", f"models/{mid}.onnx"
            out_vec = ref[vec_idx]
        else:
            spec = adaboost_to_json(model, mean, std)
            path = bundle / "models" / f"{mid}.json"
            path.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
            got = adaboost_predict_json(spec, P[:1000])
            diff = float(np.abs(got - ref[:1000]).max())
            check = {
                "max_abs_prob_diff": diff,
                "argmax_agreement": float((got.argmax(1) == ref[:1000].argmax(1)).mean()),
                "n_checked": 1000,
                "ok": diff < 1e-9,
            }
            fmt, fname = "json", f"models/{mid}.json"
            out_vec = ref[vec_idx]
        if not check["ok"]:
            raise SystemExit(f"export validation FAILED for {mid}: {check}")
        vectors["expected"][mid] = {
            "pred": out_vec.argmax(1).tolist(),
            "proba": np.round(out_vec, 6).tolist(),
        }
        manifest_models[mid] = {
            "title": TITLES[mid],
            "class": cname,
            "format": fmt,
            "file": fname,
            "size_bytes": path.stat().st_size,
            "params": model.get_params(),
            "validation": check,
            "input": "pixels",
            "output": "probabilities" if fmt == "onnx" else "vote_shares",
        }
        log(
            f"exported {mid:12s} {fmt:5s} {path.stat().st_size / 1024:9.1f} KiB   "
            f"max|dp|={check['max_abs_prob_diff']:.2e}  argmax agreement={check['argmax_agreement']:.4f}"
        )

    (bundle / "test_vectors.json").write_text(json.dumps(vectors), encoding="utf-8")
    manifest = {
        "name": "WriteRHand-MNIST",
        "version": MODEL_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "preprocessing": {
            "version": PREPROCESS_VERSION,
            "image": "28x28 grayscale, white digit on black",
            "scale": "pixel/255",
            "mean": mean,
            "std": std,
            "eps": 1e-8,
            "formula": "x = (pixel/255 - mean) / (std + eps)",
        },
        "classes": list(range(10)),
        "dataset": meta,
        "models": manifest_models,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    from src.experiments.report import build_site_data
    from src.export.modelcard import write_model_card

    site = build_site_data(Path(results_dir), manifest)
    (bundle / "results.json").write_text(json.dumps(site, separators=(",", ":")), encoding="utf-8")
    write_model_card(bundle / "README.md", manifest, site)
    log(f"bundle written to {bundle}")
    return bundle
