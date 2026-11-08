"""Assemble the deployable bundle (``artifacts/bundle``) that is uploaded to Hugging Face.

bundle/
  manifest.json                    schema ``writerhand.bundle/2``: datasets -> models -> trace contract
  README.md                        model card
  <dataset>/                       mnist | fashion_mnist | emnist_digits
    models/<id>.onnx               naive_bayes, softmax, mlp, cnn   (input pixels/255, output probabilities)
    models/<id>.params.bin         learned parameters for the execution explorer (float32, indexed in manifest)
    models/<id>.bin                SAMME ensemble: adaboost, adaboost_aug (small typed-array reader in the browser)
    python/<Class>.npz             original NumPy/torch weights (load with BaseClassifier.load)
    test_vectors.json              raw test images + expected outputs (browser/ONNX parity tests)
    results.json                   metrics used by the website dashboard

Every file is listed with its SHA-256 in the manifest, and ``bundle_id`` is a hash over all of them,
so a deployed website can state exactly which artifacts it was built against.
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.export.onnx_export import (
    adaboost_from_binary,
    adaboost_predict_binary,
    adaboost_to_binary,
    cnn_to_onnx,
    gaussian_nb_to_onnx,
    mlp_to_onnx,
    softmax_regression_to_onnx,
)
from src.export.tracespec import BUNDLE_SCHEMA, build_trace, sha256_bytes, sha256_file
from src.models.base import BaseClassifier
from src.preprocessing.preprocess import DATASETS, PREPROCESS_VERSION, load_dataset, load_raw

MODEL_VERSION = "2.0.0"

# id used in the website -> (class name, ONNX builder or None for the AdaBoost binary)
EXPORTS = {
    "naive_bayes": ("GaussianNB", gaussian_nb_to_onnx),
    "softmax": ("SoftmaxRegression", softmax_regression_to_onnx),
    "mlp": ("MLP", mlp_to_onnx),
    "cnn": ("CNN", cnn_to_onnx),
    "adaboost": ("AdaBoostSAMME", None),
}
TITLES = {
    "naive_bayes": "Gaussian Naive Bayes",
    "softmax": "Softmax regression",
    "mlp": "Multilayer perceptron",
    "cnn": "Convolutional network",
    "adaboost": "AdaBoost (SAMME)",
}
TOL = {"naive_bayes": 1e-3, "softmax": 1e-4, "mlp": 1e-4, "cnn": 1e-4}

# every model also has an augmented variant (same architecture, trained on the original images plus
# randomly rotated / shifted / scaled copies): id ``<id>_aug``, class ``<Class>_aug``
AUG_SUFFIX = "_aug"
for _mid, (_cname, _builder) in list(EXPORTS.items()):
    EXPORTS[_mid + AUG_SUFFIX] = (_cname + AUG_SUFFIX, _builder)
    TITLES[_mid + AUG_SUFFIX] = TITLES[_mid] + " (augmented)"


def base_kind(mid):
    """Model kind of an export id: ``mlp_aug`` -> ``mlp``."""
    return mid.removesuffix(AUG_SUFFIX)


def _validate_onnx(proto, P, ref_proba, tol):
    import onnxruntime as ort

    sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    outs = dict(zip(names, sess.run(names, {"pixels": P}), strict=True))
    out = outs["probabilities"]
    diff = float(np.abs(out - ref_proba).max())
    agree = float((out.argmax(1) == ref_proba.argmax(1)).mean())
    return outs, {
        "max_abs_prob_diff": diff,
        "argmax_agreement": agree,
        "n_checked": int(len(P)),
        "ok": diff <= tol and agree >= 0.999,
    }


def _check_runtime_shapes(trace, outs):
    """The manifest must describe exactly the extra outputs the graph returns."""
    for ro in trace["runtime_outputs"]:
        got = list(outs[ro["name"]].shape[1:])
        if got != ro["shape"]:
            raise SystemExit(
                f"trace shape mismatch for {ro['name']}: manifest {ro['shape']} vs graph {got}"
            )
    extra = set(outs) - {"probabilities"} - {r["name"] for r in trace["runtime_outputs"]}
    if extra:
        raise SystemExit(f"graph returns undeclared outputs: {sorted(extra)}")


def _pick_vectors(y, per_class=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.concatenate(
        [rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in np.unique(y)]
    )
    return np.sort(idx)


def export_dataset(bundle, name, artifacts_dir, results_dir, log=print):
    """Write ``bundle/<name>/...`` and return the dataset entry of the manifest."""
    import onnx

    spec = DATASETS[name]
    root = bundle / name
    (root / "models").mkdir(parents=True)
    (root / "python").mkdir()

    _, meta = load_dataset(name=name)
    mean, std = meta["mean"], meta["std"]
    _, _, Xte_raw, yte = load_raw(name)
    P = (Xte_raw.reshape(-1, 784).float() / 255.0).numpy()
    y = yte.numpy()
    Xn = (P.astype(np.float64) - mean) / (std + 1e-8)
    vec_idx = _pick_vectors(y)
    vectors = {
        "dataset": name,
        "indices": vec_idx.tolist(),
        "labels": y[vec_idx].tolist(),
        "pixels": (P[vec_idx] * 255).round().astype(int).tolist(),
        "expected": {},
    }
    check_idx = np.arange(0, len(P), max(1, len(P) // 2000))[:2000]  # ONNX parity subset
    models, files = {}, {}
    for mid, (cname, builder) in EXPORTS.items():
        npz = Path(artifacts_dir) / cname / "model.npz"
        if not npz.is_file():
            log(f"[{name}] skip {mid}: {npz} not found")
            continue
        kind = base_kind(mid)
        model = BaseClassifier.load(npz)
        shutil.copy(npz, root / "python" / f"{cname}.npz")
        ref = model.predict_proba(Xn)
        extra = None
        if kind == "adaboost":
            blob = adaboost_to_binary(model, mean, std)
            path = root / "models" / f"{mid}.bin"
            path.write_bytes(blob)
            parsed = adaboost_from_binary(blob)
            got = adaboost_predict_binary(parsed, P[check_idx])
            diff = float(np.abs(got - ref[check_idx]).max())
            check = {
                "max_abs_prob_diff": diff,
                "argmax_agreement": float((got.argmax(1) == ref[check_idx].argmax(1)).mean()),
                "n_checked": int(len(check_idx)),
                "ok": diff < 1e-9,
            }
            fmt, fname = "binary", f"models/{mid}.bin"
            extra = {
                "n_estimators": len(parsed["alphas"]),
                "node_counts": parsed["node_counts"],
                "depths": parsed["depths"],
            }
        else:
            proto = builder(model, mean, std)
            path = root / "models" / f"{mid}.onnx"
            onnx.save(proto, path)
            outs, check = _validate_onnx(
                proto, P[check_idx].astype(np.float32), ref[check_idx], TOL[kind]
            )
            fmt, fname = "onnx", f"models/{mid}.onnx"
        trace, pack = build_trace(kind, model, mean, std, spec["classes"], extra)
        if fmt == "onnx":
            _check_runtime_shapes(trace, outs)
        if not check["ok"]:
            raise SystemExit(f"export validation FAILED for {mid} on {name}: {check}")
        params = None
        if pack.entries:
            blob = pack.blob()
            ppath = root / "models" / f"{mid}.params.bin"
            ppath.write_bytes(blob)
            params = {
                "file": f"models/{mid}.params.bin",
                "sha256": sha256_bytes(blob),
                "nbytes": len(blob),
                "tensors": pack.entries,
            }
            files[f"{name}/models/{mid}.params.bin"] = params["sha256"]
        trace["params"] = params
        digest = sha256_file(path)
        files[f"{name}/{fname}"] = digest
        out_vec = ref[vec_idx]
        vectors["expected"][mid] = {
            "pred": out_vec.argmax(1).tolist(),
            "proba": np.round(out_vec, 6).tolist(),
        }
        if fmt == "onnx":
            vectors["expected"][mid]["logits"] = _logits_for(model, Xn[vec_idx], kind)
        replay = _copy_replay(
            Path(artifacts_dir) / "replay" / f"{cname}.json", root, mid, name, files
        )
        models[mid] = {
            "title": TITLES[mid],
            "kind": kind,
            "augmented": mid.endswith(AUG_SUFFIX),
            "augmentation": _augmentation_of(results_dir, cname)
            if mid.endswith(AUG_SUFFIX)
            else None,
            "class": cname,
            "format": fmt,
            "file": fname,
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "params": model.get_params(),
            "validation": check,
            "input": "pixels",
            "output": "probabilities" if fmt == "onnx" else "vote_shares",
            "trace": trace,
            "replay": replay,
        }
        log(
            f"[{name}] exported {mid:12s} {fmt:6s} {path.stat().st_size / 1024:9.1f} KiB   "
            f"max|dp|={check['max_abs_prob_diff']:.2e}  argmax agreement={check['argmax_agreement']:.4f}"
        )
    names = [m["file"] for m in models.values()]
    if len(names) != len(set(names)):
        raise SystemExit(f"[{name}] two models share one file: {sorted(names)}")
    vp = root / "test_vectors.json"
    vp.write_text(json.dumps(vectors), encoding="utf-8")
    files[f"{name}/test_vectors.json"] = sha256_file(vp)

    from src.experiments.report import build_site_data

    entry = {
        "title": spec["title"],
        "short": spec["short"],
        "kind": spec["kind"],
        "source": spec["source"],
        "classes": spec["classes"],
        "preprocessing": {
            "version": PREPROCESS_VERSION,
            "image": "28x28 grayscale, bright object on dark background",
            "scale": "pixel/255",
            "mean": mean,
            "std": std,
            "eps": 1e-8,
            "formula": "x = (pixel/255 - mean) / (std + eps)",
        },
        "dataset": meta,
        "test_vectors": f"{name}/test_vectors.json",
        "results": f"{name}/results.json",
        "models": models,
    }
    site = build_site_data(Path(results_dir), entry)
    rp = root / "results.json"
    rp.write_text(json.dumps(site, separators=(",", ":")), encoding="utf-8")
    files[f"{name}/results.json"] = sha256_file(rp)
    return entry, site, files


def _copy_replay(src, root, mid, dataset, files):
    """Copy a recorded training replay into the bundle (after checking it) and return its manifest ref."""
    if not src.is_file():
        return None
    rep = json.loads(src.read_text(encoding="utf-8"))
    if (
        rep["deployed_epoch"] is None
        or rep["schema"] != "writerhand.replay/1"
        or rep["model"] != base_kind(mid)
        or rep["dataset"] != dataset
    ):
        raise SystemExit(f"replay {src} does not belong to {dataset}/{mid}")
    n = len(rep["samples"]["indices"])
    for sn in rep["snapshots"]:
        if len(sn["proba"]) != n:
            raise SystemExit(
                f"replay {src}: snapshot {sn['epoch']} has {len(sn['proba'])} rows for {n} samples"
            )
    for k, v in rep["series"].items():
        if len(v) != len(rep["epochs"]):
            raise SystemExit(
                f"replay {src}: series {k} has {len(v)} values for {len(rep['epochs'])} epochs"
            )
    dst = root / "replay" / f"{mid}.replay.json"
    dst.parent.mkdir(exist_ok=True)
    shutil.copy(src, dst)
    digest = sha256_file(dst)
    files[f"{dataset}/replay/{mid}.replay.json"] = digest
    return {"file": f"replay/{mid}.replay.json", "sha256": digest, "size_bytes": dst.stat().st_size}


def _augmentation_of(results_dir, cname):
    """Augmentation recipe recorded in ``results/final/<Class>.json`` of an augmented variant."""
    path = Path(results_dir) / "final" / f"{cname}.json"
    return (
        json.loads(path.read_text(encoding="utf-8")).get("augmentation") if path.is_file() else None
    )


def _logits_for(model, Xn, mid):
    """Reference logits (float64) of the Python model for the expected-output vectors."""
    if mid == "naive_bayes":
        return np.round(model._joint_log_likelihood(Xn), 4).tolist()
    if mid == "softmax":
        return np.round(model._logits(Xn), 5).tolist()
    return np.round(model._get_logits(Xn).numpy().astype(np.float64), 5).tolist()


def export_all(sources, bundle_root, log=print):
    """``sources``: ``{dataset: (artifacts_dir, results_dir)}``. Writes ``<bundle_root>/bundle``."""
    bundle = Path(bundle_root) / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    datasets, sites, files = {}, {}, {}
    for name, (ad, rd) in sources.items():
        entry, site, f = export_dataset(bundle, name, ad, rd, log)
        if entry["models"]:
            datasets[name] = entry
            sites[name] = site
            files.update(f)
    if not datasets:
        raise SystemExit("nothing to export - train the models first.")
    bundle_id = sha256_bytes(json.dumps(files, sort_keys=True).encode("utf-8"))[:16]
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "name": "WriteRHand",
        "version": MODEL_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "bundle_id": bundle_id,
        "default_dataset": "mnist" if "mnist" in datasets else next(iter(datasets)),
        "files_sha256": files,
        "datasets": datasets,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    from src.export.modelcard import write_model_card

    write_model_card(bundle / "README.md", manifest, sites)
    log(f"bundle {bundle_id} written to {bundle}")
    return bundle
