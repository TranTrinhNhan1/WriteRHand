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
    """Columns: original and every transform at training and stress strength; rows: source digits."""
    data, _ = load_dataset()
    rows, _y = example_grid(data["X_train"], data["y_train"], n=3)
    ncol = len(rows)
    fig, axes = plt.subplots(3, ncol, figsize=(ncol * 0.95, 3 * 0.95 + 0.45))
    for c, (title, imgs) in enumerate(rows):
        for r in range(3):
            ax = axes[r, c]
            ax.imshow(
                imgs[r].reshape(28, 28),
                cmap="gray_r",
                vmin=float(imgs.min()),
                vmax=float(imgs.max()),
            )
            ax.axis("off")
        label = title.replace(" (stress)", "\n(kiểm tra)").replace(" (", "\n(")
        label = label.replace("\n(kiểm tra)", "\n(stress)")
        axes[0, c].set_title(label.replace("_", " "), fontsize=7.5)
    fig.subplots_adjust(wspace=0.05, hspace=0.04)
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
        for xx, t in zip(x, keys, strict=True):
            b.text(
                xx + (k - 0.5) * 0.38,
                100 * rows[i]["robust"][t] + 0.6,
                f"{100 * rows[i]['robust'][t]:.1f}",
                ha="center",
                fontsize=6,
                rotation=90,
            )
    b.set_xticks(x)
    b.set_xticklabels(
        ["clean", *[PRETTY[t].replace(" ", "\n", 1) for t in TRANSFORMS]], fontsize=7.5
    )
    b.set_ylim(60, 104)
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
    num = lambda v, f: format(v, f)  # noqa: E731
    rows = [
        ("Số tham số", lambda x: f"{x['n_parameters']:,}".replace(",", "\\,")),
        ("Accuracy test (\\%)", lambda x: num(100 * x["accuracy"]["test"], ".2f")),
        ("Accuracy train $-$ test (điểm)", lambda x: num(100 * x["generalization_gap"], ".2f")),
        ("Thời gian huấn luyện (s)", lambda x: num(x["train_time_sec"], ".0f")),
        ("Độ trễ một ảnh (ms)", lambda x: num(x["latency_ms"], ".3f")),
    ]
    body = [f"{lab} & {fn(r['mlp'])} & {fn(r['cnn'])} \\\\" for lab, fn in rows]
    m = cmp["mcnemar"]
    tex = [
        r"\begin{table}[tbp]\centering\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & MLP & CNN \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{CNN và MLP trên MNIST, cùng giao thức; mọi số lấy từ các lần chạy đã ghi. Accuracy khi ảnh test bị biến đổi nằm ở \\cref{{fig:cnn-vs-mlp}}b. "
        f"Kiểm định McNemar chính xác trên dự đoán test: "
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
        r"\begin{table}[tbp]\centering\small",
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
        r"\begin{table}[tbp]\centering\small",
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


def tex_slide_datasets(per_ds, rd, out_dir):
    """``slide_datasets.tex``: tabular for the slides (test accuracy per dataset + MNIST->EMNIST column)."""
    cross = _read(Path(rd) / "datasets" / "emnist_digits" / "cross_eval.json")
    cross_acc = {r["model"]: r["accuracy"] for r in cross[0]["rows"]} if cross else {}
    names = list(per_ds)
    head = "Mô hình & " + " & ".join(DATASETS[n]["title"] for n in names) + " & MNIST $\\to$ EMNIST"
    rows = []
    for m in ORDER:
        cells = []
        for n in names:
            f = per_ds[n][1][m]["final"]
            cells.append("--" if not f else f"{100 * f['splits']['test']['accuracy']:.2f}")
        c = cross_acc.get(m)
        cells.append("--" if c is None else f"{100 * c:.2f}")
        rows.append(f"{TITLE[m]} & " + " & ".join(cells) + " \\\\")
    tex = [
        r"\begin{tabular}{l" + "r" * (len(names) + 1) + "}",
        r"\toprule",
        head + " \\\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (Path(out_dir) / "slide_datasets.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- cost / trade-off
def _final_rows(per_ds_mnist):
    rows = []
    for mid in ORDER:
        fin = (per_ds_mnist.get(mid) or {}).get("final")
        if not fin:
            continue
        sp = fin["splits"]
        rows.append(
            {
                "id": mid,
                "fit": fin["train_time_sec"],
                "lat": fin["latency_ms_single_sample"],
                "train": 100 * sp["train"]["accuracy"],
                "val": 100 * sp["val"]["accuracy"],
                "test": 100 * sp["test"]["accuracy"],
                "predict10k": sp["test"].get("predict_time_sec"),
            }
        )
    return rows


def fig_tradeoff(plt, per_ds_mnist, out):
    """Test accuracy against training time and per-image latency, and the train-test gap."""
    rows = _final_rows(per_ds_mnist)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    a, b, c = axes
    for ax, key, xl in (
        (a, "fit", "training time on 50 000 images (s, log)"),
        (b, "lat", "latency per image (ms, log)"),
    ):
        for r in rows:
            ax.scatter(r[key], r["test"], s=46, color=COLOR[r["id"]], zorder=3)
            ax.annotate(
                TITLE[r["id"]],
                (r[key], r["test"]),
                textcoords="offset points",
                xytext=(5, -9 if r["id"] in ("softmax", "adaboost") else 4),
                fontsize=8,
            )
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("test accuracy (%)")
        ax.set_ylim(78, 100)
    a.set_xlim(0.2, 4000)
    b.set_xlim(0.004, 120)
    a.set_title("accuracy against training cost")
    b.set_title("accuracy against inference cost")
    _panel(a, "a")
    _panel(b, "b")
    x = np.arange(len(rows))
    gaps = [r["train"] - r["test"] for r in rows]
    c.bar(x, gaps, color=[COLOR[r["id"]] for r in rows], width=0.6)
    for k, g in enumerate(gaps):
        c.text(k, g + (0.1 if g >= 0 else -0.45), f"{g:+.2f}", ha="center", fontsize=8)
    c.axhline(0, color="black", lw=0.8)
    c.set_xticks(x)
    c.set_xticklabels(
        [
            {
                "naive_bayes": "NB",
                "softmax": "Softmax",
                "mlp": "MLP",
                "adaboost": "AdaBoost",
                "cnn": "CNN",
            }[r["id"]]
            for r in rows
        ],
        fontsize=8.5,
    )
    c.set_ylabel("train accuracy $-$ test accuracy (points)")
    c.set_title("generalisation gap")
    c.set_ylim(-2, 4)
    _panel(c, "c")
    fig.tight_layout()
    _save(fig, out, "tradeoff")
    plt.close(fig)


def tex_cost(per_ds_mnist, out_dir):
    """One compact table: training time, prediction time, latency and exported file size."""
    sizes = onnx_sizes()
    key = {
        "naive_bayes": "naive_bayes",
        "softmax": "softmax",
        "mlp": "mlp",
        "adaboost": "adaboost",
        "cnn": "cnn",
    }
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Mô hình & Huấn luyện (s) & Dự đoán $10\\,000$ ảnh (s) & Một ảnh (ms) & Tệp xuất (KiB) \\\\",
        "\\midrule",
    ]
    for r in _final_rows(per_ds_mnist):
        size = sizes.get(key[r["id"]])
        size_s = "--" if not size else f"{size / 1024:,.0f}".replace(",", "\\,")
        pred = "--" if r["predict10k"] is None else f"{r['predict10k']:.3f}"
        lines.append(f"{TITLE[r['id']]} & {r['fit']:.1f} & {pred} & {r['lat']:.3f} & {size_s} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (Path(out_dir) / "cost.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- key sweeps + EDA statistic
KEY_SWEEPS = {
    "naive_bayes": "var_smoothing",
    "softmax": "learning_rate",
    "mlp": "hidden_sizes",
    "adaboost": "max_depth",
    "cnn": "channels",
}


def _fmt_value(v):
    if isinstance(v, (list, tuple)):
        return "(" + ",".join(str(x) for x in v) + ")"
    if isinstance(v, float) and v != 0 and (abs(v) < 0.01 or abs(v) >= 1000):
        return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return str(v)


def fig_key_sweeps(plt, per_ds_mnist, out):
    """Train and validation accuracy against the most influential hyper-parameter of each model."""
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.0))
    flat = list(axes.ravel())
    for ax, mid, letter in zip(flat, ORDER, "abcde", strict=False):
        tune = (per_ds_mnist.get(mid) or {}).get("tuning")
        param = KEY_SWEEPS[mid]
        trials = [t for t in tune["trials"] if t.get("stage") == f"sweep:{param}"]
        xs = np.arange(len(trials))
        tr = [100 * t["train_acc"] for t in trials]
        va = [100 * t["val_acc"] for t in trials]
        ax.plot(xs, tr, "o--", color="#777777", ms=4, label="train")
        ax.plot(xs, va, "o-", color=COLOR[mid], ms=4.5, lw=1.8, label="validation")
        best = tune["best_params"][param]
        for k, t in enumerate(trials):
            if t["params"][param] == best:
                ax.scatter([k], [va[k]], s=150, facecolors="none", edgecolors="black", zorder=4)
        ax.set_xticks(xs)
        many = len(xs) >= 6
        ax.set_xticklabels(
            [_fmt_value(t["value"]) for t in trials],
            fontsize=7.5,
            rotation=45 if many else 0,
            ha="right" if many else "center",
        )
        ax.set_xlabel(param.replace("_", " "))
        ax.set_ylabel("accuracy (%)")
        ax.set_title(TITLE[mid])
        ax.legend(fontsize=8, loc="best")
        _panel(ax, letter)
    flat[5].axis("off")
    fig.tight_layout()
    _save(fig, out, "key_sweeps")
    plt.close(fig)


def tex_eda_stats(out_dir):
    """Within-class correlation of horizontally adjacent pixels (supports the Naive Bayes discussion)."""
    data, _meta = load_dataset(name="mnist")
    X, y = data["X_train"], data["y_train"]
    vals = []
    for k in range(10):
        Xk = X[y == k].reshape(-1, 28, 28)
        a = Xk[:, :, :-1].reshape(len(Xk), -1)
        b = Xk[:, :, 1:].reshape(len(Xk), -1)
        a = a - a.mean(0)
        b = b - b.mean(0)
        den = np.sqrt((a * a).mean(0) * (b * b).mean(0))
        ok = den > 1e-6
        vals.append(((a * b).mean(0)[ok] / den[ok]).mean())
    macros = [
        "% generated by src/experiments/extras_report.py -- do not edit",
        f"\\newcommand{{\\corrAdj}}{{{float(np.mean(vals)):.2f}}}",
    ]
    (Path(out_dir) / "eda_stats.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")


def tex_aug_summary(rd, out_dir):
    """Appendix table: validation delta and clean test accuracy for every augmentation condition."""
    studies = {k: _read(rd / f"augmentation_{k}.json") for k in ("mlp", "cnn")}
    if not all(studies.values()):
        return
    names = {
        "baseline": "Không tăng cường",
        "rotation": "Xoay",
        "translation": "Dịch chuyển",
        "scaling": "Co giãn",
        "gaussian_noise": "Nhiễu Gauss",
        "salt_pepper": "Nhiễu muối tiêu",
        "geometric": "Hình học (xoay, dịch, co giãn)",
        "all": "Cả năm phép",
    }
    lines = [
        r"\begin{table}[tbp]\centering\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{MLP (3 seed)} & \multicolumn{2}{c}{CNN (2 seed)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Điều kiện & $\Delta$ val (điểm) & Test (\%) & $\Delta$ val (điểm) & Test (\%) \\",
        r"\midrule",
    ]
    for key, label in names.items():
        cells = []
        for k in ("mlp", "cnn"):
            c = studies[k]["conditions"][key]
            if key == "baseline":
                cells.append("--")
            else:
                cells.append(
                    f"{100 * c['val_delta_vs_baseline_mean']:+.2f} $\\pm$ {100 * c['val_delta_vs_baseline_sd']:.2f}"
                )
            cells.append(f"{100 * c['test_acc_mean']:.2f}")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Tăng cường dữ liệu với cấu hình đã chọn: $\Delta$ là hiệu validation accuracy so với điều kiện gốc \emph{cùng seed} (trung bình $\pm$ độ lệch chuẩn mẫu của hiệu), ``Test'' là accuracy test sạch trung bình trên các seed. Mỗi điều kiện tăng cường dùng $100\,000$ ảnh train, điều kiện gốc $50\,000$.}",
        r"\label{tab:augmentation}",
        r"\end{table}",
    ]
    (Path(out_dir) / "augmentation_summary.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


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
                    tex_slide_datasets(per_ds, rd, gen),
                ),
            )
        )
    jobs.append(("cross eval", lambda: tex_cross_eval(rd, gen)))
    jobs.append(("augmentation summary", lambda: tex_aug_summary(rd, gen)))
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
    if "mnist" in per_ds:
        jobs.append(
            (
                "tradeoff",
                lambda: (
                    fig_tradeoff(plt, per_ds["mnist"][1], fig_dir),
                    tex_cost(per_ds["mnist"][1], gen),
                ),
            )
        )
    if "mnist" in per_ds:
        jobs.append(("key sweeps", lambda: fig_key_sweeps(plt, per_ds["mnist"][1], fig_dir)))
    jobs.append(("eda stats", lambda: tex_eda_stats(gen)))
    if (Path(ad) / "CNN" / "model.npz").is_file():
        jobs.append(("cnn filters", lambda: fig_cnn_filters(plt, ad, fig_dir)))
    for name, fn in jobs:
        try:
            fn()
            log(f"extras: {name} ok")
        except Exception as e:  # each figure is optional on its own
            log(f"extras: {name} skipped: {type(e).__name__}: {e}")
