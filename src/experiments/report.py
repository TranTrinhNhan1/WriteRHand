"""Tables, figures and website data built from ``results/`` (never from hand-typed numbers)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.experiments.search import CLASS_NAME, SPECS
from src.utils.paths import ARTIFACTS_DIR, ROOT

ORDER = ["naive_bayes", "softmax", "mlp", "adaboost", "cnn"]
TITLE = {
    "naive_bayes": "Gaussian NB",
    "softmax": "Softmax regression",
    "mlp": "MLP",
    "adaboost": "AdaBoost-SAMME",
    "cnn": "CNN",
}
COLOR = {
    "naive_bayes": "#8c6d31",
    "softmax": "#3b6ea8",
    "mlp": "#b5483a",
    "adaboost": "#4d8a5b",
    "cnn": "#6b4c9a",
}
SPLITS = ("train", "val", "test")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else None


def load_all(results_dir):
    rd = Path(results_dir)
    out = {}
    for mid in ORDER:
        c = CLASS_NAME[mid]
        out[mid] = {
            "final": _read(rd / "final" / f"{c}.json"),
            "tuning": _read(rd / "tuning" / f"{c}.json"),
            "sklearn": _read(rd / "baselines" / f"sklearn_{c}.json"),
        }
    return out


def top_confusions(cm, k=6):
    cm = np.array(cm)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    return [
        {"true": int(i // 10), "pred": int(i % 10), "count": int(off[i // 10, i % 10])}
        for i in flat
        if off[i // 10, i % 10] > 0
    ]


def _split_summary(s):
    keys = (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "predict_time_sec",
        "predict_time_ms_per_sample",
    )
    d = {k: s[k] for k in keys if k in s}
    d["confusion_matrix"] = s["confusion_matrix"]
    return d


def _tuning_entry(t):
    """Sweeps of one ``results/tuning/<Class>.json`` in the shape the website draws."""
    sweeps = {}
    for tr in t["trials"]:
        if tr["param"] is None:
            continue
        sweeps.setdefault(tr["param"], []).append(
            {
                "value": tr["value"],
                "train_acc": tr["train_acc"],
                "val_acc": tr["val_acc"],
                "train_time_sec": tr["train_time_sec"],
                "stage": tr["stage"],
            }
        )
    return {
        "best_params": t["best_params"],
        "n_trials": t["n_trials"],
        "sweeps": sweeps,
        "epoch_curve": t.get("epoch_curve"),
        "curve": next((tr["curve"] for tr in t["trials"] if tr["stage"].startswith("curve")), None),
    }


# models measured in results/ but not exported to the browser bundle (no ONNX, no trace)
EXTRA_MODELS = (("svm", "LinearSVM", "Linear SVM"), ("stacking", "StackingEnsemble", "Stacking"))
EXTRA_SCHEMA = "writerhand.extra/1"


def _final_entry(mid, cname, title, f, sk):
    entry = {
        "id": mid,
        "name": title,
        "class": cname,
        "params": f["params"],
        "train_time_sec": f["train_time_sec"],
        "latency_ms_single_sample": f["latency_ms_single_sample"],
        "splits": {s: _split_summary(f["splits"][s]) for s in SPLITS},
        "top_confusions": top_confusions(f["splits"]["test"]["confusion_matrix"]),
        "history": f.get("history") or {},
        "best_epoch": f.get("best_epoch"),
        "stopped_epoch": f.get("stopped_epoch"),
        "in_browser": False,
        "sklearn": None,
    }
    if sk:
        entry["sklearn"] = {
            "class": sk["sklearn_class"],
            "train_time_sec": sk["train_time_sec"],
            "latency_ms_single_sample": sk["latency_ms_single_sample"],
            "splits": {
                s: {k: sk["splits"][s][k] for k in ("accuracy", "f1_macro", "predict_time_sec")}
                for s in SPLITS
            },
        }
    return entry


def build_extra_models(results_dir, browser_ids=()):
    """Website data of the models that are measured but not in the browser bundle (linear SVM, stacking).

    Only numbers read from ``results_dir`` (``final/``, ``tuning/``, ``baselines/``); ``None`` when neither
    model has a final result. The output is deterministic (no timestamp), so a regenerated file only
    differs when the results do. ``browser_ids``: model ids of this dataset in the browser bundle (the SVM is).
    """
    rd = Path(results_dir)
    cls_to_id = {**{v: k for k, v in CLASS_NAME.items()}, **{c: m for m, c, _ in EXTRA_MODELS}}
    models, tuning = [], {}
    for mid, cname, title in EXTRA_MODELS:
        f = _read(rd / "final" / f"{cname}.json")
        if not f:
            continue
        entry = _final_entry(
            mid, cname, title, f, _read(rd / "baselines" / f"sklearn_{cname}.json")
        )
        entry["in_browser"] = mid in set(browser_ids)
        if mid == "stacking":
            bases = f.get("base_models") or f["params"].get("bases") or []
            base_fit = [
                (_read(rd / "final" / f"{b}.json") or {}).get("train_time_sec") for b in bases
            ]
            meta_fit = f.get("meta_fit_sec", f["train_time_sec"])
            entry["stacking"] = {
                "fitted_on": "val",
                "meta_learner": "softmax",
                "n_features": f.get("n_features"),
                "meta_fit_sec": meta_fit,
                "base_fit_sec": sum(base_fit) if all(x is not None for x in base_fit) else None,
                "selected_l2": f.get("selected_l2"),
                "cv_val_acc": f.get("cv_val_acc"),
                "soft_vote_test_accuracy": f.get("soft_vote_test_accuracy"),
                "bases": [
                    {
                        "id": cls_to_id.get(b, b),
                        "test_accuracy": (f.get("base_test_accuracy") or {}).get(b),
                        "weight": (f.get("block_importance") or {}).get(b),
                    }
                    for b in bases
                ],
            }
            # total cost: the six base models are trained first, then the meta-learner
            if entry["stacking"]["base_fit_sec"] is not None:
                entry["train_time_sec"] = entry["stacking"]["base_fit_sec"] + meta_fit
            cv = f.get("cv_on_validation") or {}
            if cv:
                pts = sorted(((float(k), v) for k, v in cv.items()), key=lambda p: p[0])
                tuning[mid] = {
                    "best_params": {"l2": f.get("selected_l2")},
                    "n_trials": len(pts),
                    "sweeps": {
                        "l2": [
                            {"value": k, "val_acc": v["cv_val_acc_mean"], "stage": "cv:l2"}
                            for k, v in pts
                        ]
                    },
                    "epoch_curve": None,
                    "curve": None,
                }
        else:
            t = _read(rd / "tuning" / f"{cname}.json")
            if t:
                tuning[mid] = _tuning_entry(t)
        models.append(entry)
    out = {"models": models, "tuning": tuning} if models else {}
    pairs = build_pairs(rd)
    if pairs:
        out["pairs"] = pairs
    return out or None


PAIR_KEYS = (
    "train_acc",
    "val_acc",
    "test_acc",
    "test_f1_macro",
    "fit_time_sec",
    "latency_ms",
    "test_rotation_acc",
    "test_stress_acc",
)


def build_pairs(results_dir):
    """Plain vs augmented variant of every model (``augmentation_pairs.json``), for the website; ``None`` if absent.

    The recipe is the augmented variants' own record, completed with the default range of every listed
    operation that the record does not override (``src.experiments.augment.DEFAULTS``).
    """
    raw = _read(Path(results_dir) / "augmentation_pairs.json")
    if not raw or not raw.get("models"):
        return None
    from src.experiments.augment import DEFAULTS

    recipe = None
    models = {}
    for kind, pair in raw["models"].items():
        if not pair.get("plain") or not pair.get("augmented"):
            continue
        models[kind] = {
            tag: {k: pair[tag].get(k) for k in PAIR_KEYS} for tag in ("plain", "augmented")
        }
        aug = pair["augmented"].get("augmentation")
        if aug and recipe is None:
            ranges = aug.get("ranges") or {}
            recipe = {
                "ops": list(aug.get("ops") or []),
                "copies": aug.get("k"),
                "ranges": {op: ranges.get(op, DEFAULTS.get(op)) for op in aug.get("ops") or []},
            }
    if not models:
        return None
    return {"rotation_angles": raw.get("rotation_angles"), "recipe": recipe, "models": models}


def write_extra_models(datasets, out_path, manifest=None):
    """Write ``extra_models.json`` for the site: ``datasets`` maps a dataset id to its results dir.

    ``manifest`` (the site's bundle manifest) tells which of these models also run in the browser.
    """
    out = {"schema": EXTRA_SCHEMA, "datasets": {}}
    for name, rd in datasets.items():
        ids = list(((manifest or {}).get("datasets", {}).get(name) or {}).get("models", {}))
        e = build_extra_models(rd, ids)
        if e:
            out["datasets"][name] = e
    Path(out_path).write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    return out


def build_site_data(results_dir, manifest=None):
    """JSON consumed by the website dashboard (only measured numbers)."""
    data = load_all(results_dir)
    models, tuning = [], {}
    any_final = next((v["final"] for v in data.values() if v["final"]), None)
    for mid in ORDER:
        f, t, sk = data[mid]["final"], data[mid]["tuning"], data[mid]["sklearn"]
        if not f:
            continue
        entry = {
            "id": mid,
            "name": TITLE[mid],
            "class": CLASS_NAME[mid],
            "params": f["params"],
            "train_time_sec": f["train_time_sec"],
            "latency_ms_single_sample": f["latency_ms_single_sample"],
            "splits": {s: _split_summary(f["splits"][s]) for s in SPLITS},
            "top_confusions": top_confusions(f["splits"]["test"]["confusion_matrix"]),
            "history": f.get("history", {}),
            "best_epoch": f.get("best_epoch"),
            "stopped_epoch": f.get("stopped_epoch"),
            "in_browser": bool(manifest and mid in manifest["models"]),
            "sklearn": None,
        }
        if sk:
            entry["sklearn"] = {
                "class": sk["sklearn_class"],
                "train_time_sec": sk["train_time_sec"],
                "latency_ms_single_sample": sk["latency_ms_single_sample"],
                "splits": {
                    s: {k: sk["splits"][s][k] for k in ("accuracy", "f1_macro", "predict_time_sec")}
                    for s in SPLITS
                },
            }
        models.append(entry)
        if t:
            tuning[mid] = _tuning_entry(t)
    extras = {}
    for key, fname in (
        ("cnn_vs_mlp", "cnn_vs_mlp.json"),
        ("cross_eval", "cross_eval.json"),
    ):
        v = _read(Path(results_dir) / fname)
        if v:
            extras[key] = v
    if "cnn_vs_mlp" in extras and manifest:
        for row in extras["cnn_vs_mlp"]["rows"]:
            m = manifest["models"].get(row["id"])
            if m:
                row["onnx_bytes"] = m["size_bytes"]
    aug = {k: _read(Path(results_dir) / f"augmentation_{k}.json") for k in ("mlp", "cnn")}
    aug = {k: v for k, v in aug.items() if v}
    if aug:
        extras["augmentation"] = aug
    return {
        **extras,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": any_final["dataset"] if any_final else None,
        "hardware": any_final["hardware"] if any_final else None,
        "git": any_final["git"] if any_final else None,
        "protocol": (
            "Train 50 000 / validation 10 000 (stratified split of the official training "
            "set) / official test set. Hyper-parameters selected on validation only; "
            "test evaluated for the final configurations."
        ),
        "models": models,
        "tuning": tuning,
        "manifest": manifest,
    }


# --------------------------------------------------------------------------- tables
def benchmark_rows(data):
    rows = []
    for mid in ORDER:
        f = data[mid]["final"]
        if not f:
            continue
        s = f["splits"]
        rows.append(
            {
                "model": TITLE[mid],
                "id": mid,
                "acc_train": s["train"]["accuracy"],
                "acc_val": s["val"]["accuracy"],
                "acc_test": s["test"]["accuracy"],
                "prec_test": s["test"]["precision_macro"],
                "rec_test": s["test"]["recall_macro"],
                "f1_test": s["test"]["f1_macro"],
                "train_time": f["train_time_sec"],
                "predict_time_test": s["test"]["predict_time_sec"],
                "latency_ms": f["latency_ms_single_sample"],
            }
        )
    return rows


def extra_rows(results_dir):
    """Rows of the models added after the first five (linear SVM, stacking), if their results exist."""
    rows = []
    for title, cname in (("Linear SVM", "LinearSVM"), ("Stacking (6 models)", "StackingEnsemble")):
        f = _read(Path(results_dir) / "final" / f"{cname}.json")
        if not f:
            continue
        s = f["splits"]
        rows.append(
            {
                "model": title,
                "id": cname,
                "acc_train": s["train"]["accuracy"],
                "acc_val": s["val"]["accuracy"],
                "acc_test": s["test"]["accuracy"],
                "prec_test": s["test"]["precision_macro"],
                "rec_test": s["test"]["recall_macro"],
                "f1_test": s["test"]["f1_macro"],
                "train_time": f["train_time_sec"],
                "predict_time_test": s["test"]["predict_time_sec"],
                "latency_ms": f["latency_ms_single_sample"],
            }
        )
    return rows


def write_tables(data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = benchmark_rows(data)
    md_rows = rows + extra_rows(out_dir.parent)
    head = [
        "model",
        "acc_train",
        "acc_val",
        "acc_test",
        "prec_test",
        "rec_test",
        "f1_test",
        "train_time",
        "predict_time_test",
        "latency_ms",
    ]
    (out_dir / "benchmark.csv").write_text(
        ",".join(head)
        + "\n"
        + "\n".join(
            ",".join(str(r[h]) if h == "model" else f"{r[h]:.6g}" for h in head) for r in rows
        )
        + "\n",
        encoding="utf-8",
    )
    md = [
        "| Model | Acc train | Acc val | Acc test | P macro | R macro | F1 macro | Fit (s) | "
        "Predict 10k (s) | 1-sample (ms) |",
        "|" + "---|" * 10,
    ]
    for r in md_rows:
        md.append(
            f"| {r['model']} | {100 * r['acc_train']:.2f} | {100 * r['acc_val']:.2f} | "
            f"{100 * r['acc_test']:.2f} | {100 * r['prec_test']:.2f} | {100 * r['rec_test']:.2f} | "
            f"{100 * r['f1_test']:.2f} | {r['train_time']:.1f} | {r['predict_time_test']:.3f} | "
            f"{r['latency_ms']:.3f} |"
        )
    (out_dir / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # LaTeX fragments (\input in the report)
    tex = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Mô hình & Train & Val & Test & P$_{\text{macro}}$ & R$_{\text{macro}}$ & F1$_{\text{macro}}$ \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"{r['model']} & {100 * r['acc_train']:.2f} & {100 * r['acc_val']:.2f} & "
            f"\\textbf{{{100 * r['acc_test']:.2f}}} & {100 * r['prec_test']:.2f} & "
            f"{100 * r['rec_test']:.2f} & {100 * r['f1_test']:.2f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "benchmark_accuracy.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    tex = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Mô hình & Huấn luyện (s) & Dự đoán 10\,000 mẫu (s) & 1 mẫu (ms) \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"{r['model']} & {r['train_time']:.1f} & {r['predict_time_test']:.3f} & "
            f"{r['latency_ms']:.3f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "benchmark_time.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # ours vs scikit-learn
    tex = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Mô hình & Test (ours) & Test (sklearn) & $|\Delta|$ & Fit ours (s) & Fit sklearn (s) \\",
        r"\midrule",
    ]
    for mid in ORDER:
        f, sk = data[mid]["final"], data[mid]["sklearn"]
        if f and sk:
            a, b = f["splits"]["test"]["accuracy"], sk["splits"]["test"]["accuracy"]
            tex.append(
                f"{TITLE[mid]} & {100 * a:.2f} & {100 * b:.2f} & {100 * abs(a - b):.2f} & "
                f"{f['train_time_sec']:.1f} & {sk['train_time_sec']:.1f} \\\\"
            )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "sklearn_comparison.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # tuning summary
    tex = [
        r"\begin{tabular}{@{}lp{9.0cm}r@{}}",
        r"\toprule",
        r"Mô hình & Siêu tham số được chọn & Số cấu hình đã chạy \\",
        r"\midrule",
    ]
    for mid in ORDER:
        t = data[mid]["tuning"]
        if t:
            bp = ", ".join(
                f"{k}={tuple(v) if isinstance(v, list) else v}"
                for k, v in t["best_params"].items()
                if k not in ("seed", "verbose")
            )
            tex.append(
                f"{TITLE[mid]} & \\texttt{{{bp.replace('_', chr(92) + '_')}}} & {t['n_trials']} \\\\"
            )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "tuning_summary.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- figures
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Latin Modern Roman (same family as the LaTeX report) when the TeX distribution ships it
    lm_dir = Path("/usr/share/texmf/fonts/opentype/public/lm")
    serif = ["DejaVu Serif"]
    for f in ("lmroman10-regular", "lmroman10-bold", "lmroman10-italic", "lmroman10-bolditalic"):
        fp = lm_dir / f"{f}.otf"
        if fp.is_file():
            font_manager.fontManager.addfont(str(fp))
            serif = ["Latin Modern Roman", *serif]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif,
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "legend.frameon": False,
            "axes.axisbelow": True,
            "axes.unicode_minus": False,
        }
    )
    return plt


def _plain_log(ax, axis="y"):
    """Log axis with plain decimal tick labels (0.01, 0.1, 1, 10) instead of 10^k."""
    from matplotlib.ticker import FuncFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _panel(ax, letter):
    ax.text(
        -0.02,
        1.08,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")


CLASS_NAMES = [str(i) for i in range(10)]
LABEL_WORD = "digit"


def set_dataset_labels(names, word):
    """Class names used on confusion-matrix axes (set per dataset by ``build_report``)."""
    global CLASS_NAMES, LABEL_WORD
    CLASS_NAMES, LABEL_WORD = list(names), word


def _class_ticks(ax, with_y=True):
    names = CLASS_NAMES
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    if all(len(n) == 1 for n in names):
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
    else:
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(names if with_y else [""] * len(names), fontsize=7)


def fig_confusions(plt, data, out):
    mids = [m for m in ORDER if data[m]["final"]]
    fig, axes = plt.subplots(1, len(mids), figsize=(3.5 * len(mids) + 0.6, 3.9))
    axes = np.atleast_1d(axes)
    im = None
    for k, (ax, mid) in enumerate(zip(axes, mids, strict=True)):
        cm = np.array(data[mid]["final"]["splits"]["test"]["confusion_matrix"], dtype=float)
        norm = cm / cm.sum(1, keepdims=True)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(10):
            for j in range(10):
                if cm[i, j] > 0:
                    ax.text(
                        j,
                        i,
                        int(cm[i, j]),
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        color="white" if norm[i, j] > 0.55 else "#333",
                    )
        acc = data[mid]["final"]["splits"]["test"]["accuracy"]
        ax.set_title(f"{TITLE[mid]}\ntest accuracy {100 * acc:.2f}%")
        _class_ticks(ax, k == 0)
        ax.set_xlabel("predicted " + LABEL_WORD)
        if k == 0:
            ax.set_ylabel("true " + LABEL_WORD)
        _panel(ax, "abcde"[k])
        _save_single(plt, mid, cm, norm, out)
    fig.subplots_adjust(wspace=0.18)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.012, pad=0.012)
    cb.set_label("share of the true digit (row-normalised)")
    _save(fig, out, "confusion_all")
    plt.close(fig)


def _save_single(plt, mid, cm, norm, out):
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if norm[i, j] > 0.55 else "#333",
                )
    _class_ticks(ax, True)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_title(f"{TITLE[mid]} - test confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    _save(fig, out, f"confusion_{mid}")
    plt.close(fig)


def sweep_points(tuning):
    """``{param: [trial, ...]}`` for every one-factor sweep, in the order of the tested values.

    ``tune`` re-uses the record of a configuration that was already fitted (the incumbent value
    is part of every sweep but is fitted only once), so the incumbent of a sweep lives in an
    earlier stage. Here it is put back into the sweep it belongs to: a trial belongs to the sweep
    of ``param`` when it differs from the sweep's reference configuration in ``param`` only.
    """
    sweeps = {}
    trials = tuning["trials"]
    for tr in trials:
        if tr["param"] and tr["stage"].startswith("sweep:"):
            sweeps.setdefault(tr["param"], []).append(tr)
    out = {}
    for param, trs in sweeps.items():
        ref = {k: v for k, v in trs[0]["params"].items() if k != param}
        have = {json.dumps(t["value"]) for t in trs}
        pts = list(trs)
        for tr in trials:
            rest = {k: v for k, v in tr["params"].items() if k != param}
            v = tr["params"].get(param)
            if (
                rest == ref
                and json.dumps(v) not in have
                and tr["stage"].startswith(("sweep:", "base"))
            ):
                pts.append({**tr, "value": v})
                have.add(json.dumps(v))
        order = {
            json.dumps(list(v) if isinstance(v, tuple) else v): i
            for i, v in enumerate(dict(SPECS[_mid_of(tuning)]["sweeps"])[param])
        }
        pts.sort(key=lambda t: order.get(json.dumps(t["value"]), 1e9))
        out[param] = pts
    return out


def _mid_of(tuning):
    return next(m for m, c in CLASS_NAME.items() if c == tuning["model"])


def fig_sweeps(plt, data, out):
    from matplotlib.gridspec import GridSpec

    for mid in ORDER:
        t = data[mid]["tuning"]
        if not t:
            continue
        by = sweep_points(t)
        if not by or t.get("smoke"):
            continue
        n = len(by)
        cols = 2 if n == 4 else min(n, 3)
        rows = (n + cols - 1) // cols
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        fig = plt.figure(figsize=(3.7 * cols, 4.1 * rows))
        gs = GridSpec(rows, cols, hspace=0.5, wspace=0.34)
        first = None
        for idx, (param, trs) in enumerate(by.items()):
            r, c = divmod(idx, cols)
            sub = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[r, c], height_ratios=[3, 1.05], hspace=0.14
            )
            ax = fig.add_subplot(sub[0])
            axt = fig.add_subplot(sub[1], sharex=ax)
            vals = [tr["value"] for tr in trs]
            numeric = all(isinstance(v, (int, float)) for v in vals)
            x = np.array(vals, dtype=float) if numeric else np.arange(len(vals))
            if numeric and len(x) > 2:  # crowded linear axis (e.g. 0.9 / 0.95): use even spacing
                gaps = np.diff(np.sort(x))
                if gaps.min() / (x.max() - x.min()) < 0.08 and not (
                    min(vals) > 0 and max(vals) / min(vals) >= 20
                ):
                    x_labels, x = [f"{v:g}" for v in vals], np.arange(len(vals), dtype=float)
                else:
                    x_labels = [f"{v:g}" for v in vals]
            else:
                x_labels = [f"{v:g}" for v in vals] if numeric else None
            va = np.array([100 * tr["val_acc"] for tr in trs])
            ta = np.array([100 * tr["train_acc"] for tr in trs])
            ax.fill_between(x, va, ta, color="#bbb", alpha=0.25, lw=0, label="generalisation gap")
            (l1,) = ax.plot(x, ta, "o--", color="#777", ms=3.5, lw=1, label="train")
            (l2,) = ax.plot(x, va, "o-", color=COLOR[mid], ms=4.5, lw=1.8, label="validation")
            best = int(np.argmax(va))
            ax.annotate(
                f"{va[best]:.2f}%",
                (x[best], va[best]),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=8,
                color=COLOR[mid],
                fontweight="bold",
            )
            ax.plot(x[best], va[best], "o", ms=10, mfc="none", mec="k", lw=1)
            log = numeric and min(vals) > 0 and max(vals) / min(vals) >= 20
            for a in (ax, axt):
                if log:
                    a.set_xscale("log")
            ax.set_ylabel("accuracy (%)")
            plt.setp(ax.get_xticklabels(), visible=False)
            axt.bar(
                x if not log else x,
                [tr["train_time_sec"] for tr in trs],
                width=(0.6 if not numeric else None) or None,
                color="#999",
                alpha=0.8,
            ) if not numeric else axt.plot(
                x, [tr["train_time_sec"] for tr in trs], "s-", color="#777", ms=3, lw=1
            )
            axt.set_ylabel("fit (s)")
            axt.set_yscale("log")
            times = [tr["train_time_sec"] for tr in trs]
            axt.set_yticks(
                [10 ** np.floor(np.log10(min(times))), 10 ** np.ceil(np.log10(max(times)))]
            )
            _plain_log(axt)
            axt.set_xlabel(param.replace("_", " "))
            if numeric:
                axt.set_xticks(x)
                axt.set_xticklabels(
                    x_labels,
                    rotation=0 if len(x) <= 6 else 45,
                    ha="center" if len(x) <= 6 else "right",
                )
                axt.minorticks_off()
            else:
                axt.set_xticks(x)
                axt.set_xticklabels(
                    [str(tuple(v)) if isinstance(v, list) else str(v) for v in vals],
                    rotation=40,
                    ha="right",
                    fontsize=7.5,
                )
            _panel(ax, "abcdef"[idx])
            first = first or ax
        first.legend(loc="lower right", fontsize=8)
        fig.suptitle(
            f"{TITLE[mid]}: one-factor sweeps on the validation split "
            "(circle = selected value; lower strip = fit time)",
            y=1.0,
            fontsize=10.5,
        )
        _save(fig, out, f"sweep_{mid}")
        plt.close(fig)


def fig_learning_curves(plt, data, out):
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    f = data["softmax"]["final"]
    if f:
        h = f["history"]
        ax = axes[0]
        e = range(1, len(h["loss"]) + 1)
        ax.plot(e, h["loss"], color=COLOR["softmax"], label="train loss (+ L2 term)")
        if h.get("val_loss"):
            ax.plot(
                range(1, len(h["val_loss"]) + 1),
                h["val_loss"],
                "--",
                color="#444",
                label="validation loss",
            )
        ax.set_title("Softmax regression")
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        _panel(ax, "a")
    f = data["mlp"]["final"]
    if f:
        h = f["history"]
        ax = axes[1]
        ax.plot(
            range(1, len(h["loss"]) + 1),
            h["loss"],
            color=COLOR["mlp"],
            label="train loss (dropout on)",
        )
        if h.get("inner_val_loss"):
            ax.plot(
                range(1, len(h["inner_val_loss"]) + 1),
                h["inner_val_loss"],
                "--",
                color="#444",
                label="early-stopping hold-out",
            )
        if h.get("val_loss"):
            ax.plot(
                range(1, len(h["val_loss"]) + 1),
                h["val_loss"],
                ":",
                color="#1b6ca8",
                lw=1.6,
                label="validation (monitor only)",
            )
        if f.get("best_epoch"):
            ax.axvline(f["best_epoch"], color="k", lw=0.8, ls=":")
            ax.text(
                f["best_epoch"],
                0.04,
                f"restored epoch {f['best_epoch']} ",
                transform=ax.get_xaxis_transform(),
                ha="right",
                fontsize=8,
            )
        ax.set_yscale("log")
        ax.set_title("MLP")
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy (log)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _plain_log(ax)
        ax.legend(loc="upper right", fontsize=8)
        _panel(ax, "b")
    f = data["adaboost"]["final"]
    if f:
        h = f["history"]
        ax = axes[2]
        ax.plot(
            range(1, len(h["train_acc"]) + 1),
            [100 * v for v in h["train_acc"]],
            color="#888",
            label="train",
        )
        if h.get("val_acc"):
            ax.plot(
                range(1, len(h["val_acc"]) + 1),
                [100 * v for v in h["val_acc"]],
                color=COLOR["adaboost"],
                label="validation",
            )
        ax.set_title("AdaBoost-SAMME")
        ax.set_xlabel("boosting rounds $T$")
        ax.set_ylabel("accuracy (%)")
        ax.legend(loc="lower right")
        _panel(ax, "c")
    fig.tight_layout()
    _save(fig, out, "learning_curves")
    plt.close(fig)


def fig_benchmark(plt, data, out):
    rows = benchmark_rows(data)
    if not rows:
        return
    names = [
        r["model"].replace(" regression", "\nregression").replace("-SAMME", "\nSAMME") for r in rows
    ]
    cols = [COLOR[r["id"]] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))

    # (a) error rate, log scale (accuracy near 100% hides differences on a linear axis)
    ax = axes[0]
    w = 0.27
    shades = ["#c9c9c9", "#8f8f8f", "#222222"]
    for k, (key, lab) in enumerate(
        (("acc_train", "train"), ("acc_val", "validation"), ("acc_test", "test"))
    ):
        err = [100 * (1 - r[key]) for r in rows]
        bars = ax.bar(x + (k - 1) * w, err, w, label=lab, color=shades[k])
        if lab == "test":
            for b, v in zip(bars, err, strict=True):
                ax.text(
                    b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.1f}", ha="center", fontsize=7.5
                )
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50])
    ax.set_ylim(top=max(100 * (1 - r["acc_test"]) for r in rows) * 1.8)
    _plain_log(ax)
    ax.set_ylabel("error rate (%, log scale)")
    ax.set_title("Error rate = 100 \u2212 accuracy")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.52), ncol=3, fontsize=8, columnspacing=1)
    _panel(ax, "a")

    def bars_log(ax, key, title, ylabel, letter, fmt):
        vals = [r[key] for r in rows]
        bs = ax.bar(x, vals, 0.58, color=cols)
        for b, v in zip(bs, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.15, fmt(v), ha="center", fontsize=8)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(min(vals) / 2.2, max(vals) * 4)
        _plain_log(ax)
        _panel(ax, letter)

    bars_log(axes[1], "train_time", "Training time", "seconds (log)", "b", lambda v: f"{v:.3g} s")
    bars_log(
        axes[2],
        "predict_time_test",
        "Predicting 10 000 test images",
        "seconds (log)",
        "c",
        lambda v: f"{v:.3g} s",
    )
    bars_log(
        axes[3],
        "latency_ms",
        "Latency for one image",
        "milliseconds (log)",
        "d",
        lambda v: f"{v:.3g} ms",
    )
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n.replace(" ", "\n", 1).replace("-", "-\n", 1) if len(n) > 7 else n for n in names],
            fontsize=8,
        )
    fig.tight_layout()
    _save(fig, out, "benchmark")
    plt.close(fig)


def fig_top_confusions(plt, data, out):
    mids = [m for m in ORDER if data[m]["final"]]
    fig, axes = plt.subplots(1, len(mids), figsize=(3.3 * len(mids), 3.0), sharex=True)
    axes = np.atleast_1d(axes)
    for k, (ax, mid) in enumerate(zip(axes, mids, strict=True)):
        tc = top_confusions(data[mid]["final"]["splits"]["test"]["confusion_matrix"], 6)
        labels = [f"{c['true']} \u2192 {c['pred']}" for c in tc][::-1]
        counts = [c["count"] for c in tc][::-1]
        bs = ax.barh(labels, counts, color=COLOR[mid])
        for b, v in zip(bs, counts, strict=True):
            ax.text(v + 0.4, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=8)
        ax.set_title(TITLE[mid])
        ax.set_xlabel("test errors (count)")
        ax.grid(axis="y", visible=False)
        _panel(ax, "abcde"[k])
    fig.suptitle(
        "Most frequent confusions (true digit \u2192 predicted digit); same x-scale in all panels",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, out, "top_confusions")
    plt.close(fig)


def fig_interpretability(plt, data, out, artifacts_dir):
    from src.models.base import BaseClassifier

    def load(c):
        p = Path(artifacts_dir) / c / "model.npz"
        return BaseClassifier.load(p) if p.is_file() else None

    nb, lr, mlp, ada = (
        load("GaussianNB"),
        load("SoftmaxRegression"),
        load("MLP"),
        load("AdaBoostSAMME"),
    )
    fig, axes = plt.subplots(4, 10, figsize=(10, 4.6))
    rows = []
    if nb is not None:
        rows.append(("NB class mean", nb.mean.reshape(10, 28, 28), "gray"))
    if lr is not None:
        rows.append(("softmax weights", lr.W.T.reshape(10, 28, 28), "RdBu_r"))
    if mlp is not None:
        W0 = mlp.weights_numpy()[0][0]  # (784, H)
        rows.append(("MLP layer-1 units", W0[:, :10].T.reshape(10, 28, 28), "RdBu_r"))
    if ada is not None:
        imp = np.zeros(784)
        for t in range(ada.n_estimators_):
            f = ada.tree_feature[t]
            for node in np.flatnonzero(f >= 0):
                imp[f[node]] += ada.alphas[t]
        rows.append(("AdaBoost split usage", np.tile(imp.reshape(1, 28, 28), (10, 1, 1)), "magma"))
    for r, (title, imgs, cmap) in enumerate(rows):
        for c in range(10):
            ax = axes[r, c]
            im = imgs[c]
            v = np.abs(im).max() if cmap == "RdBu_r" else None
            ax.imshow(im, cmap=cmap, vmin=-v if v else None, vmax=v)
            ax.axis("off")
            if title.startswith("AdaBoost") and c > 0:
                ax.axis("off")
                ax.clear()
                ax.axis("off")
            if c == 0:
                ax.text(
                    -0.15, 0.5, title, transform=ax.transAxes, ha="right", va="center", fontsize=8
                )
            if r == 0 and not title.startswith("AdaBoost"):
                ax.set_title(str(c), fontsize=8)
    for r in range(len(rows), 4):
        for c in range(10):
            axes[r, c].axis("off")
    fig.tight_layout()
    _save(fig, out, "interpretability")
    plt.close(fig)


def fig_errors(plt, data, out, artifacts_dir):
    """Misclassified test digits of the best model."""
    from src.preprocessing.preprocess import load_mnist

    best = max(
        (m for m in ORDER if data[m]["final"]),
        key=lambda m: data[m]["final"]["splits"]["test"]["accuracy"],
        default=None,
    )
    if best is None:
        return
    pred_file = Path(data_dir_of(out)) / "final" / f"{CLASS_NAME[best]}_predictions.npz"
    if not pred_file.is_file():
        return
    pred = np.load(pred_file)["test"].astype(int)
    _, _, Xte, yte = load_mnist()
    wrong = np.flatnonzero(pred != yte.numpy())[:20]
    fig, axes = plt.subplots(2, 10, figsize=(9, 2.2))
    for ax, i in zip(axes.ravel(), wrong, strict=False):
        ax.imshow(Xte[i].numpy(), cmap="gray_r")
        ax.axis("off")
        ax.set_title(f"{int(yte[i])}→{pred[i]}", fontsize=8)
    for ax in axes.ravel()[len(wrong) :]:
        ax.axis("off")
    fig.suptitle(f"First test errors of {TITLE[best]} (true→predicted)", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "errors")
    plt.close(fig)


def data_dir_of(fig_dir):
    return Path(fig_dir).parent


def fig_eda(plt, out):
    from src.preprocessing.preprocess import load_mnist

    Xtr, ytr, _, _ = load_mnist()
    X = Xtr.numpy().astype(np.float32) / 255.0
    y = ytr.numpy()
    fig = plt.figure(figsize=(11, 3.0))
    gs = fig.add_gridspec(2, 14, hspace=0.05, wspace=0.05)
    rng = np.random.default_rng(0)
    for r in range(2):
        for c in range(10):
            ax = fig.add_subplot(gs[r, c])
            idx = rng.choice(np.flatnonzero(y == c))
            ax.imshow(X[idx], cmap="gray_r")
            ax.axis("off")
    for k in range(2):
        ax = fig.add_subplot(gs[k, 10:12])
        ax.imshow(X[y == (3 if k == 0 else 8)].mean(0), cmap="gray_r")
        ax.axis("off")
        ax.set_title("mean " + str(3 if k == 0 else 8), fontsize=7)
    ax = fig.add_subplot(gs[0, 12:])
    ax.imshow(X.std(0) == 0, cmap="gray_r")
    ax.axis("off")
    ax.set_title("std = 0 pixels", fontsize=7)
    ax = fig.add_subplot(gs[1, 12:])
    ax.imshow(X.mean(0), cmap="gray_r")
    ax.axis("off")
    ax.set_title("mean image", fontsize=7)
    _save(fig, out, "eda_overview")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    counts = np.bincount(y, minlength=10)
    ax.bar(range(10), 100 * counts / counts.sum(), color="#3b6ea8")
    ax.axhline(10, color="k", lw=0.7, ls=":")
    ax.set_xlabel("digit")
    ax.set_ylabel("% of training images")
    ax.set_xticks(range(10))
    _save(fig, out, "eda_class_balance")
    plt.close(fig)


def update_readme(readme, table_md):
    """Replace the block between the BENCHMARK markers of README.md by the measured table."""
    start, end = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"
    text = Path(readme).read_text(encoding="utf-8")
    if start not in text or end not in text:
        return False
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    Path(readme).write_text(f"{head}{start}\n{table_md.strip()}\n{end}{tail}", encoding="utf-8")
    return True


def build_report(results_dir, artifacts_dir=ARTIFACTS_DIR, log=print, dataset="mnist"):
    from src.preprocessing.preprocess import DATASETS

    results_dir = Path(results_dir)
    data = load_all(results_dir)
    set_dataset_labels(
        DATASETS[dataset]["classes"], "digit" if DATASETS[dataset]["kind"] == "digits" else "class"
    )
    plt = _style()
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    write_tables(data, results_dir / "tables")
    if results_dir.resolve() == (ROOT / "results").resolve():
        update_readme(ROOT / "README.md", (results_dir / "tables" / "benchmark.md").read_text())
        from src.experiments.texfrag import write_fragments

        write_fragments(results_dir, ROOT / "report" / "generated", artifacts_dir)
    fig_confusions(plt, data, fig_dir)
    fig_sweeps(plt, data, fig_dir)
    fig_learning_curves(plt, data, fig_dir)
    fig_benchmark(plt, data, fig_dir)
    fig_top_confusions(plt, data, fig_dir)
    if dataset != "mnist":
        log(f"[{dataset}] digit-specific figures (interpretability, errors, eda) skipped")
        log(f"tables -> {results_dir / 'tables'} ; figures -> {fig_dir}")
        return data
    for name, fn in (
        ("interpretability", lambda: fig_interpretability(plt, data, fig_dir, artifacts_dir)),
        ("errors", lambda: fig_errors(plt, data, fig_dir, artifacts_dir)),
        ("eda", lambda: fig_eda(plt, fig_dir)),
    ):
        try:  # these need the raw data / saved models, so each is optional on its own
            fn()
        except Exception as e:
            log(f"skipped figure '{name}': {type(e).__name__}: {e}")
    from src.experiments.extras_report import build_extras

    build_extras(log=log)
    log(f"tables -> {results_dir / 'tables'} ; figures -> {fig_dir}")
    return data


__all__ = ["build_report", "build_site_data", "SPECS", "ROOT"]
