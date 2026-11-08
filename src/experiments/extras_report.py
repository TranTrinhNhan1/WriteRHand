"""Figures and tables of the advanced extensions (datasets, augmentation, CNN) from ``results/``.

Output goes to ``results/figures`` and ``results/tables`` (MNIST top-level) and is read by the
report. Every number is read from the JSON files written by the experiments.
"""

import json
from pathlib import Path

import numpy as np

from src.experiments.augment import TRANSFORMS, example_grid
from src.experiments.report import COLOR, ORDER, TITLE, _panel, _save, _style, load_all
from src.models.base import BaseClassifier
from src.preprocessing.preprocess import DATASETS, load_dataset
from src.utils.paths import ROOT, dataset_dirs

PRETTY = {
    "rotation": "rotation",
    "translation": "translation",
    "scaling": "scaling",
    "gaussian_noise": "Gaussian noise",
    "salt_pepper": "salt & pepper",
    "geometric": "geometric (3)",
    "all": "all five",
    "baseline": "baseline",
}


def _read(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


# --------------------------------------------------------------------------- augmentation
def fig_augmentation_examples(plt, out):
    data, _ = load_dataset()
    rows, y = example_grid(data["X_train"], data["y_train"], n=8)
    fig, axes = plt.subplots(len(rows), 8, figsize=(8 * 0.95, len(rows) * 0.95 + 0.2))
    for r, (title, imgs) in enumerate(rows):
        for c in range(8):
            ax = axes[r, c]
            ax.imshow(
                imgs[c].reshape(28, 28),
                cmap="gray_r",
                vmin=float(imgs.min()),
                vmax=float(imgs.max()),
            )
            ax.axis("off")
        axes[r, 0].text(
            -0.15, 0.5, title, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=7.5
        )
    fig.subplots_adjust(left=0.27, wspace=0.04, hspace=0.04)
    _save(fig, out, "augmentation_examples")
    plt.close(fig)


def fig_augmentation(plt, study, out, dataset_note=""):
    conds = study["conditions"]
    names = [n for n in conds if n != "baseline"]
    names.sort(key=lambda n: conds[n]["val_delta_vs_baseline_mean"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    d = np.array([100 * conds[n]["val_delta_vs_baseline_mean"] for n in names])
    sd = np.array([100 * conds[n]["val_delta_vs_baseline_sd"] for n in names])
    cols = [
        "#3b6ea8" if v - s > 0 else "#b5483a" if v + s < 0 else "#8a8a8a"
        for v, s in zip(d, sd, strict=True)
    ]
    y = np.arange(len(names))
    a.barh(y, d, xerr=sd, color=cols, capsize=3, height=0.62, ecolor="#222")
    a.axvline(0, color="k", lw=0.8)
    a.set_yticks(y)
    a.set_yticklabels([PRETTY[n] for n in names])
    a.set_xlabel("validation accuracy minus baseline (points, mean ± sd over paired seeds)")
    a.set_title(f"{study['model'].upper()}: effect on validation accuracy{dataset_note}")
    _panel(a, "a")
    pert = list(TRANSFORMS)
    order = ["baseline", *names[::-1]]
    M = np.array([[100 * conds[n]["robust_test_acc_mean"][p] for p in pert] for n in order])
    im = b.imshow(M, cmap="Blues", vmin=M.min() - 3, vmax=100, aspect="auto")
    b.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            b.text(
                j,
                i,
                f"{M[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if M[i, j] > 0.5 * (M.max() + M.min()) + 2 else "#222",
            )
    b.set_xticks(range(len(pert)))
    b.set_xticklabels([PRETTY[p].replace(" ", "\n", 1) for p in pert], fontsize=8)
    b.set_yticks(range(len(order)))
    b.set_yticklabels([PRETTY[n] for n in order])
    b.set_xlabel("stress perturbation of the test images (report only)")
    b.set_title("test accuracy (%) under perturbation")
    fig.colorbar(im, ax=b, fraction=0.04, pad=0.02)
    _panel(b, "b")
    fig.tight_layout()
    _save(fig, out, f"augmentation_{study['model']}")
    plt.close(fig)


def table_augmentation(study, out_dir):
    conds = study["conditions"]
    lines = [
        "| condition | n_train | val acc % | test acc % | Δ val vs baseline (pts) | fit (s) |",
        "|---|---|---|---|---|---|",
    ]
    for n, c in conds.items():
        t = np.mean([r["fit_time_sec"] for r in c["runs"]])
        lines.append(
            f"| {PRETTY[n]} | {c['runs'][0]['n_train']} | {100 * c['val_acc_mean']:.2f} ± {100 * c['val_acc_std']:.2f} | "
            f"{100 * c['test_acc_mean']:.2f} ± {100 * c['test_acc_std']:.2f} | "
            f"{100 * c.get('val_delta_vs_baseline_mean', 0):+.2f} ± {100 * c.get('val_delta_vs_baseline_sd', 0):.2f} | {t:.0f} |"
        )
    (out_dir / f"augmentation_{study['model']}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- datasets
def collect_datasets():
    out = {}
    for name in DATASETS:
        rd, _ = dataset_dirs(name)
        data = load_all(rd)
        if any(data[m]["final"] for m in ORDER):
            out[name] = (rd, data)
    return out


def fig_datasets(plt, per_ds, out):
    names = list(per_ds)
    mids = [m for m in ORDER if any(per_ds[n][1][m]["final"] for n in names)]
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    w = 0.8 / len(mids)
    x = np.arange(len(names))
    for k, m in enumerate(mids):
        acc = [
            100 * per_ds[n][1][m]["final"]["splits"]["test"]["accuracy"]
            if per_ds[n][1][m]["final"]
            else np.nan
            for n in names
        ]
        bars = ax.bar(
            x + (k - (len(mids) - 1) / 2) * w, acc, w * 0.92, color=COLOR[m], label=TITLE[m]
        )
        for r, v in zip(bars, acc, strict=True):
            if not np.isnan(v):
                ax.text(
                    r.get_x() + r.get_width() / 2,
                    v + 0.4,
                    f"{v:.1f}",
                    ha="center",
                    fontsize=6.5,
                    rotation=90,
                    va="bottom",
                )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASETS[n]["title"] for n in names])
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(60, 104)
    ax.legend(ncol=len(mids), loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8)
    _save(fig, out, "datasets_accuracy")
    plt.close(fig)


def table_datasets(per_ds, out_dir):
    names = list(per_ds)
    hdr = "| Model | " + " | ".join(f"{DATASETS[n]['title']} acc / F1" for n in names) + " |"
    lines = [hdr, "|---|" + "---|" * len(names)]
    for m in ORDER:
        cells = []
        for n in names:
            f = per_ds[n][1][m]["final"]
            cells.append(
                "n/a"
                if not f
                else f"{100 * f['splits']['test']['accuracy']:.2f} / {100 * f['splits']['test']['f1_macro']:.2f}"
            )
        lines.append(f"| {TITLE[m]} | " + " | ".join(cells) + " |")
    (out_dir / "datasets_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- CNN
def fig_cnn_vs_mlp(plt, cmp, per_ds_mnist, out):
    rows = {r["id"]: r for r in cmp["rows"]}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    a, b, c = axes
    ids = ["mlp", "cnn"]
    a.bar(
        [TITLE[i] for i in ids],
        [rows[i]["n_parameters"] for i in ids],
        color=[COLOR[i] for i in ids],
        width=0.55,
    )
    a.set_yscale("log")
    for k, i in enumerate(ids):
        a.text(
            k,
            rows[i]["n_parameters"] * 1.1,
            f"{rows[i]['n_parameters']:,}",
            ha="center",
            fontsize=8,
        )
    a.set_ylabel("parameters (log)")
    a.set_ylim(top=rows["mlp"]["n_parameters"] * 4)
    a.set_title("model size")
    _panel(a, "a")
    keys = ["clean", *TRANSFORMS]
    x = np.arange(len(keys))
    for k, i in enumerate(ids):
        b.bar(
            x + (k - 0.5) * 0.38,
            [100 * rows[i]["robust"][t] for t in keys],
            0.36,
            color=COLOR[i],
            label=TITLE[i],
        )
    b.set_xticks(x)
    b.set_xticklabels(
        ["clean", *[PRETTY[t].replace(" ", "\n", 1) for t in TRANSFORMS]], fontsize=7.5
    )
    b.set_ylim(60, 100)
    b.set_ylabel("test accuracy (%)")
    b.set_title("accuracy under perturbation (no augmentation)")
    b.legend(fontsize=8)
    _panel(b, "b")
    for i in ids:
        h = per_ds_mnist[i]["final"]["history"]
        c.plot(
            range(1, len(h["val_acc"]) + 1),
            [100 * v for v in h["val_acc"]],
            color=COLOR[i],
            label=TITLE[i],
            marker="o",
            ms=3,
        )
    c.set_xlabel("epoch")
    c.set_ylabel("validation accuracy (%)")
    c.set_ylim(94, 100)
    c.set_title("learning curves")
    c.legend(fontsize=8)
    _panel(c, "c")
    fig.tight_layout()
    _save(fig, out, "cnn_vs_mlp")
    plt.close(fig)


def fig_cnn_filters(plt, artifacts_dir, out):
    """The learned first-layer kernels and the feature maps of one real test digit."""
    model = BaseClassifier.load(Path(artifacts_dir) / "CNN" / "model.npz")
    data, _ = load_dataset()
    x = data["X_test"][:1]
    tr = model.forward_trace(x)
    K = model.K[0].numpy()[:, 0]
    n = len(K)
    cols = min(n, 16)
    fig, axes = plt.subplots(3, cols, figsize=(cols * 0.8, 3 * 0.85 + 0.4))
    for j in range(cols):
        m = np.abs(K[j]).max()
        axes[0, j].imshow(K[j], cmap="RdBu_r", vmin=-m, vmax=m)
        axes[1, j].imshow(tr["act0"][0, j], cmap="gray_r")
        axes[2, j].imshow(tr["pool0"][0, j], cmap="gray_r")
        for r in range(3):
            axes[r, j].axis("off")
    for r, t in enumerate(["kernels (3×3)", "ReLU maps", "after pooling"]):
        axes[r, 0].text(
            -0.2, 0.5, t, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8
        )
    fig.subplots_adjust(left=0.13, wspace=0.08, hspace=0.08)
    _save(fig, out, "cnn_filters")
    plt.close(fig)


def onnx_sizes():
    """ONNX file sizes (bytes) of the exported MNIST bundle, ``{model: bytes}`` (empty if not exported)."""
    from src.utils.paths import ARTIFACTS_DIR

    p = ARTIFACTS_DIR / "bundle" / "manifest.json"
    if not p.is_file():
        return {}
    ds = json.loads(p.read_text(encoding="utf-8"))["datasets"].get("mnist", {})
    return {k: m.get("size_bytes") for k, m in ds.get("models", {}).items()}


def table_cnn_vs_mlp(cmp, out_dir):
    lines = [
        "| | MLP | CNN |",
        "|---|---|---|",
    ]
    r = {x["id"]: x for x in cmp["rows"]}

    def row(label, fn):
        lines.append(f"| {label} | {fn(r['mlp'])} | {fn(r['cnn'])} |")

    row("parameters", lambda x: f"{x['n_parameters']:,}")
    sizes = onnx_sizes()
    row(
        "ONNX size (KiB)",
        lambda x: f"{(sizes.get(x['id']) or 0) / 1024:.0f}" if sizes.get(x["id"]) else "n/a",
    )
    row("test accuracy %", lambda x: f"{100 * x['accuracy']['test']:.2f}")
    row(
        "macro P / R / F1 %",
        lambda x: (
            f"{100 * x['precision_macro']:.2f} / {100 * x['recall_macro']:.2f} / {100 * x['f1_macro']:.2f}"
        ),
    )
    row("train − test accuracy (pts)", lambda x: f"{100 * x['generalization_gap']:.2f}")
    row("training time (s)", lambda x: f"{x['train_time_sec']:.0f}")
    row("1-image latency (ms)", lambda x: f"{x['latency_ms']:.3f}")
    for t in ("clean", *TRANSFORMS):
        row(f"acc. under {t} %", lambda x, t=t: f"{100 * x['robust'][t]:.2f}")
    (out_dir / "cnn_vs_mlp.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_cnn_vs_mlp(cmp, out_dir):
    """``cnn_vs_mlp.tex``: the comparison table of the report (every value read from the JSON)."""
    r = {x["id"]: x for x in cmp["rows"]}
    sizes = onnx_sizes()
    num = lambda v, f: format(v, f)  # noqa: E731
    rows = [
        ("Số tham số", lambda x: f"{x['n_parameters']:,}".replace(",", "\\,")),
        (
            "Kích thước ONNX (KiB)",
            lambda x: f"{sizes[x['id']] / 1024:.0f}" if sizes.get(x["id"]) else "--",
        ),
        ("Accuracy test (\\%)", lambda x: num(100 * x["accuracy"]["test"], ".2f")),
        (
            "Precision / Recall / F1 macro (\\%)",
            lambda x: (
                f"{100 * x['precision_macro']:.2f} / {100 * x['recall_macro']:.2f} / {100 * x['f1_macro']:.2f}"
            ),
        ),
        ("Accuracy train $-$ test (điểm)", lambda x: num(100 * x["generalization_gap"], ".2f")),
        ("Thời gian huấn luyện (s)", lambda x: num(x["train_time_sec"], ".0f")),
        ("Độ trễ một ảnh (ms)", lambda x: num(x["latency_ms"], ".3f")),
    ]
    names = {
        "rotation": "xoay",
        "translation": "dịch",
        "scaling": "co giãn",
        "gaussian_noise": "nhiễu Gauss",
        "salt_pepper": "muối tiêu",
    }
    rows += [
        (f"Accuracy khi {names[t]} (\\%)", lambda x, t=t: num(100 * x["robust"][t], ".2f"))
        for t in TRANSFORMS
    ]
    body = [f"{lab} & {fn(r['mlp'])} & {fn(r['cnn'])} \\\\" for lab, fn in rows]
    m = cmp["mcnemar"]
    tex = [
        r"\begin{table}[H]\centering\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & MLP & CNN \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{CNN và MLP trên MNIST, cùng giao thức; mọi số lấy từ các lần chạy đã ghi. Hàng cuối: ảnh test bị biến đổi cố định "
        f"(seed 999, mức mạnh hơn lúc huấn luyện), chỉ để báo cáo. Kiểm định McNemar chính xác trên dự đoán test: "
        f"chỉ MLP đúng {m['only_mlp']} ảnh, chỉ CNN đúng {m['only_cnn']} ảnh, $p={m['p']:.2g}$.}}",
        r"\label{tab:cnn-vs-mlp}",
        r"\end{table}",
    ]
    (out_dir / "cnn_vs_mlp.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def tex_datasets(per_ds, out_dir):
    """``datasets.tex``: test accuracy / macro-F1 of every model on every dataset that was trained."""
    names = list(per_ds)
    rows = []
    for m in ORDER:
        cells = []
        for n in names:
            f = per_ds[n][1][m]["final"]
            cells.append(
                "--"
                if not f
                else f"{100 * f['splits']['test']['accuracy']:.2f} / {100 * f['splits']['test']['f1_macro']:.2f}"
            )
        rows.append(f"{TITLE[m]} & " + " & ".join(cells) + " \\\\")
    tex = [
        r"\begin{table}[H]\centering\small",
        r"\begin{tabular}{l" + "r" * len(names) + "}",
        r"\toprule",
        "Mô hình & " + " & ".join(DATASETS[n]["title"] for n in names) + " \\\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Accuracy / macro-F1 trên tập test (\%) của từng bộ dữ liệu; mỗi bộ dữ liệu có phép chia train/validation/test riêng, siêu tham số chọn riêng trên validation của bộ đó (quét rút gọn cho Fashion-MNIST và EMNIST Digits). Không so sánh chéo giữa các bộ dữ liệu.}",
        r"\label{tab:datasets}",
        r"\end{table}",
    ]
    (out_dir / "datasets.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def tex_cross_eval(rd, out_dir):
    """``cross_eval.tex``: models trained on MNIST evaluated on the EMNIST Digits test set (labelled as such)."""
    data = _read(Path(rd) / "datasets" / "emnist_digits" / "cross_eval.json")
    if not data:
        return
    item = data[0]
    own = {}
    own_rd = Path(rd) / "datasets" / "emnist_digits"
    for r in item["rows"]:
        cname = {
            "naive_bayes": "GaussianNB",
            "softmax": "SoftmaxRegression",
            "mlp": "MLP",
            "cnn": "CNN",
            "adaboost": "AdaBoostSAMME",
        }[r["model"]]
        fin = _read(own_rd / "final" / f"{cname}.json")
        own[r["model"]] = fin["splits"]["test"]["accuracy"] if fin else None
    rows = []
    for r in item["rows"]:
        own_acc = "--" if own[r["model"]] is None else f"{100 * own[r['model']]:.2f}"
        rows.append(f"{TITLE[r['model']]} & {100 * r['accuracy']:.2f} & {own_acc} \\\\")
    tex = [
        r"\begin{table}[H]\centering\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Mô hình & Huấn luyện trên MNIST & Huấn luyện trên EMNIST Digits \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Accuracy (\%) trên $40\,000$ ảnh test của EMNIST Digits. Cột giữa là mô hình huấn luyện trên MNIST (dùng hằng số chuẩn hóa của MNIST) đánh giá trực tiếp; cột phải là mô hình huấn luyện trên EMNIST Digits. Đây là hai mô hình khác nhau.}",
        r"\label{tab:cross-eval}",
        r"\end{table}",
    ]
    (out_dir / "cross_eval.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    pre = {"naive_bayes": "NB", "softmax": "SM", "mlp": "MLP", "cnn": "CNN", "adaboost": "ADA"}
    macros = ["% generated by src/experiments/extras_report.py -- do not edit"]
    for r in item["rows"]:
        macros.append(f"\\newcommand{{\\cross{pre[r['model']]}}}{{{100 * r['accuracy']:.2f}}}")
        if own[r["model"]] is not None:
            macros.append(
                f"\\newcommand{{\\ownEmnist{pre[r['model']]}}}{{{100 * own[r['model']]:.2f}}}"
            )
    (out_dir / "cross_eval_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")


def build_extras(log=print):
    plt = _style()
    rd, ad = dataset_dirs("mnist")
    fig_dir, tab_dir = rd / "figures", rd / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    jobs.append(("augmentation examples", lambda: fig_augmentation_examples(plt, fig_dir)))
    for k in ("mlp", "cnn"):
        study = _read(rd / f"augmentation_{k}.json")
        if study:
            jobs.append(
                (
                    f"augmentation {k}",
                    lambda s=study: (
                        fig_augmentation(plt, s, fig_dir),
                        table_augmentation(s, tab_dir),
                    ),
                )
            )
    per_ds = collect_datasets()
    gen = ROOT / "report" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    if len(per_ds) > 1:
        jobs.append(
            (
                "datasets",
                lambda: (
                    fig_datasets(plt, per_ds, fig_dir),
                    table_datasets(per_ds, tab_dir),
                    tex_datasets(per_ds, gen),
                ),
            )
        )
    jobs.append(("cross eval", lambda: tex_cross_eval(rd, gen)))
    cmp = _read(rd / "cnn_vs_mlp.json")
    if cmp and "mnist" in per_ds:
        jobs.append(
            (
                "cnn vs mlp",
                lambda: (
                    fig_cnn_vs_mlp(plt, cmp, per_ds["mnist"][1], fig_dir),
                    table_cnn_vs_mlp(cmp, tab_dir),
                    tex_cnn_vs_mlp(cmp, gen),
                ),
            )
        )
    if (Path(ad) / "CNN" / "model.npz").is_file():
        jobs.append(("cnn filters", lambda: fig_cnn_filters(plt, ad, fig_dir)))
    for name, fn in jobs:
        try:
            fn()
            log(f"extras: {name} ok")
        except Exception as e:  # each figure is optional on its own
            log(f"extras: {name} skipped: {type(e).__name__}: {e}")
