"""Model card (README.md of the Hugging Face repository), generated from measured results."""

PREPROCESS = "mnist-global-standardize-v2-stratified"
DESCRIPTIONS = {
    "cnn": "two Conv-ReLU-MaxPool stages and a linear head; hand-written im2col convolution, pooling and backpropagation",
    "naive_bayes": "per-class, per-pixel Gaussian likelihoods with MLE parameters and log-space inference",
    "svm": "one-vs-rest linear SVM, hinge loss, sub-gradient mini-batch descent, L2; outputs scores, not probabilities",
    "softmax": "multiclass logistic regression, softmax + cross-entropy, mini-batch gradient descent, L2",
    "mlp": "fully connected ReLU network, hand-written backpropagation, momentum SGD, early stopping",
    "adaboost": "multiclass AdaBoost (SAMME) over weighted histogram decision trees",
}


def _rows(site):
    out = []
    for m in site["models"]:
        s = m["splits"]
        out.append(
            f"| {m['name']} | {100 * s['train']['accuracy']:.2f} | {100 * s['val']['accuracy']:.2f} | "
            f"{100 * s['test']['accuracy']:.2f} | {100 * s['test']['f1_macro']:.2f} | "
            f"{m['train_time_sec']:.1f} | {m['latency_ms_single_sample']:.3f} |"
        )
    return "\n".join(out)


def write_model_card(path, manifest, sites):
    """``sites``: ``{dataset: site_data}`` as produced by ``build_site_data``."""
    sections, files = [], []
    for ds_id, ds in manifest["datasets"].items():
        site = sites[ds_id]
        meta, pre = ds["dataset"], ds["preprocessing"]
        hw = site.get("hardware") or {}
        sections.append(
            f"""### {ds["title"]} (`{ds_id}`)
Classes: {", ".join(f"{i} = {c}" for i, c in enumerate(ds["classes"]))}.
Split: {meta["n_train"]} train / {meta["n_val"]} validation (stratified, seed {meta["seed"]},
fingerprint `{meta["split_fingerprint"]}`) / {meta["n_test"]} official test. Normalisation
`mean = {pre["mean"]:.6f}`, `std = {pre["std"]:.6f}` (training portion only). CPU:
{hw.get("cpu", "n/a")}, {hw.get("logical_cores", "?")} logical cores.

| Model | Train acc % | Val acc % | Test acc % | Test macro-F1 % | Fit (s) | 1-image latency (ms) |
|---|---|---|---|---|---|---|
{_rows(site)}
"""
        )
        for v in ds["models"].values():
            files.append(
                f"| `{ds_id}/{v['file']}` | {v['format']} | {v['size_bytes'] / 1024:.0f} KiB | "
                f"`{v['sha256'][:12]}` | {v['validation']['argmax_agreement'] * 100:.2f}% / "
                f"{v['validation']['max_abs_prob_diff']:.1e} |"
            )
    desc = "\n".join(
        f"- **{v['title']}** - {DESCRIPTIONS[k.removesuffix('_aug')]}."
        + (
            " Same architecture, trained on the original images plus randomly rotated (up to 30 degrees), "
            "shifted and scaled copies; compared side by side with the plain model."
            if v.get("augmented")
            else ""
        )
        for k, v in next(iter(manifest["datasets"].values()))["models"].items()
    )
    git = next((s.get("git") or {} for s in sites.values()), {}).get("commit")
    text = f"""---
library_name: onnx
tags:
  - image-classification
  - mnist
  - fashion-mnist
  - emnist
  - educational
  - from-scratch
datasets:
  - ylecun/mnist
  - zalando-datasets/fashion_mnist
metrics:
  - accuracy
  - f1
license: other
license_name: unspecified-educational
---

# WriteRHand

Six image classifiers (each also as an augmented variant) whose mathematical core was implemented **from scratch** (NumPy; PyTorch
only as a tensor library, *no autograd, no nn.Module, no F.conv2d*) for a university Pattern
Recognition course, trained **separately** on three 28x28 datasets: MNIST, Fashion-MNIST and
EMNIST Digits. They are *educational* models, not production OCR. Code, tests and report:
<https://github.com/TranTrinhNhan1/WriteRHand>. The live demo and the Model Execution Explorer
run these files in the browser.

{desc}

## Intended use and limitations
- A model is only meaningful for the dataset it was trained on: the Fashion-MNIST models output
  clothing classes, the MNIST/EMNIST models digit classes. No cross-dataset claim is made unless
  a result table says so explicitly.
- Drawings are cropped and centred with the MNIST recipe, but robustness to hand-drawn input is
  limited. Naive Bayes is weak by design (independence assumption). AdaBoost scores are weighted
  vote shares, **not calibrated probabilities**.

## Input / output contract (`{PREPROCESS}`)
Feed raw `pixels` in [0, 1], shape `[N, 784]` (row-major 28x28); the graph standardises
internally and returns `probabilities` `[N, 10]`. Extra optional outputs expose intermediate
tensors for the explorer; the layer graph, shapes and parameter files are described per model in
`manifest.json` (`trace`, schema `writerhand.trace/1`; see `docs/TRACE_CONTRACT.md` in the code
repository). They never change `probabilities`.

## Results (measured, CPU)
Hyper-parameters were chosen on the validation split only; test accuracy is reported once for the
final configuration. Times: wall-clock fit time on the 50 000-image train split and median latency
of a single-image `predict`.

{chr(10).join(sections)}

## Files (bundle `{manifest["bundle_id"]}`)
| file | format | size | sha256 (prefix) | exported vs Python: argmax agreement / max prob. diff |
|---|---|---|---|---|
{chr(10).join(files)}

`python/*.npz` hold the original weights (`BaseClassifier.load`), `test_vectors.json` raw test
images with expected outputs, `results.json` the benchmark numbers used by the website.

## Training procedure
See the repository: `uv run python -m src prepare | tune | train | baselines | export | upload`
(`--dataset mnist|fashion_mnist|emnist_digits`). Experiment tracking: Weights & Biases (offline
fallback). Seeds are fixed (git commit `{git}`).

## AI-assisted development
Parts of this code, the tests, the export pipeline and the website were produced with the help of an
AI coding assistant and reviewed/run by the project authors, as disclosed in the
course report.

## License
No license has been chosen by the authors yet (educational project). Datasets: MNIST (LeCun et
al.), Fashion-MNIST (Xiao et al., MIT), EMNIST (Cohen et al.).

## Citation
LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). Gradient-based learning applied to document
recognition. Xiao, H., Rasul, K., Vollgraf, R. (2017). Fashion-MNIST. Cohen, G. et al. (2017). EMNIST.
"""
    path.write_text(text, encoding="utf-8")
