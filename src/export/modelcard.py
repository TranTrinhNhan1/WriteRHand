"""Model card (README.md of the Hugging Face repository), generated from measured results."""

DESCRIPTIONS = {
    "naive_bayes": "per-class, per-pixel Gaussian likelihoods with MLE parameters and log-space inference",
    "softmax": "multiclass logistic regression, softmax + cross-entropy, mini-batch gradient descent, L2",
    "mlp": "fully connected ReLU network, hand-written backpropagation, momentum SGD, early stopping",
    "adaboost": "multiclass AdaBoost (SAMME) over weighted histogram decision trees",
}


def write_model_card(path, manifest, site):
    rows = []
    for m in site["models"]:
        s = m["splits"]
        rows.append(
            f"| {m['name']} | {100 * s['train']['accuracy']:.2f} | {100 * s['val']['accuracy']:.2f} | "
            f"{100 * s['test']['accuracy']:.2f} | {100 * s['test']['f1_macro']:.2f} | "
            f"{m['train_time_sec']:.1f} | {m['latency_ms_single_sample']:.3f} |"
        )
    files = "\n".join(
        f"| `{v['file']}` | {v['format']} | {v['size_bytes'] / 1024:.0f} KiB | "
        f"{v['validation']['argmax_agreement'] * 100:.2f}% / {v['validation']['max_abs_prob_diff']:.1e} | "
        f"{v['validation']['n_checked']:,} |"
        for v in manifest["models"].values()
    )
    hw = site.get("hardware") or {}
    ds = manifest["dataset"]
    pre = manifest["preprocessing"]
    desc = "\n".join(
        f"- **{v['title']}** - {DESCRIPTIONS[k]}." for k, v in manifest["models"].items()
    )
    text = f"""---
library_name: onnx
tags:
  - image-classification
  - mnist
  - educational
  - from-scratch
datasets:
  - ylecun/mnist
metrics:
  - accuracy
  - f1
license: other
license_name: unspecified-educational
---

# WriteRHand-MNIST

Four handwritten-digit classifiers whose mathematical core was implemented **from scratch**
(NumPy; PyTorch only as a tensor library for the MLP, no autograd) for a university Pattern
Recognition course. They are *educational* models, not production OCR.
Code, tests and report: <https://github.com/TranTrinhNhan1/WriteRHand>.
Live demo (runs the models in the browser): see the repository README.

{desc}

## Intended use and limitations
- Recognise a single, centred, 28x28 handwritten digit (white on black), MNIST style.
- Not trained on other scripts, multiple digits, noisy photographs or non-centred input; the demo
  therefore crops and centres drawings with the MNIST recipe, but robustness is limited.
- Accuracy numbers below are for MNIST only. Naive Bayes is weak by design (independence
  assumption). AdaBoost scores are weighted vote shares, **not calibrated probabilities**.

## Data and preprocessing (`{pre["version"]}`)
- MNIST: 60 000 training + 10 000 test images. Training set split with a seeded **stratified**
  split into {ds["n_train"]} train / {ds["n_val"]} validation (split fingerprint `{ds["split_fingerprint"]}`,
  seed {ds["seed"]}); the official 10 000 test images are used only for final reporting.
- Pixels / 255, then `x = (p - mean) / (std + eps)` with `mean = {pre["mean"]:.6f}`,
  `std = {pre["std"]:.6f}`, `eps = 1e-8` (computed on the training portion only). The ONNX graphs
  contain this step: feed raw `pixels` in [0, 1], shape `[N, 784]`, read `probabilities` `[N, 10]`.
  The ONNX files also expose optional extra outputs (hidden activations, weights, class means and
  variances) that the website draws in its "look inside" panel; they never change `probabilities`.

## Results (measured, CPU)
Hyper-parameters were chosen on the validation split only. Time columns: wall-clock fit time on
the full 50 000-image train split and median latency of a single-image `predict`
(CPU: {hw.get("cpu", "n/a")}, {hw.get("logical_cores", "?")} logical cores, PyTorch threads: {hw.get("torch_threads", "?")}).

| Model | Train acc % | Val acc % | Test acc % | Test macro-F1 % | Fit (s) | 1-image latency (ms) |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Files
| file | format | size | exported vs Python: argmax agreement / max prob. diff | test images checked |
|---|---|---|---|---|
{files}

`python/*.npz` hold the original weights (`BaseClassifier.load`), `test_vectors.json` raw test
digits with expected outputs, `results.json` all benchmark numbers used by the website.

## Training procedure
See the repository: `uv run python -m src prepare | tune | train | baselines | export | upload`.
Experiment tracking: Weights & Biases (offline fallback). Seeds are fixed; runs are reproducible on
the same hardware/library versions (git commit `{(site.get("git") or {}).get("commit")}`).

## AI-assisted development
Parts of this code, the tests, the export pipeline and the website were produced with the help of an
AI coding assistant (Claude Code) and reviewed/run by the project authors, as disclosed in the
course report.

## License
No license has been chosen by the authors yet (educational project). MNIST is provided by
Y. LeCun, C. Cortes and C. J. C. Burges.

## Citation
LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). Gradient-based learning applied to
document recognition. *Proceedings of the IEEE* 86(11), 2278-2324.
"""
    path.write_text(text, encoding="utf-8")
