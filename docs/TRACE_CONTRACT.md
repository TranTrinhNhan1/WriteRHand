# Trace contract (`writerhand.trace/1`)

The **Model Execution Explorer** of the website draws only what this contract describes. Python
generates it from the *fitted model object* (`src/export/tracespec.py`); the frontend validates it
(`web/src/engine/contract.ts`) and refuses to draw a model whose files do not match their hashes or
whose shapes disagree with the graph. Nothing about depth, widths, kernel sizes, channel counts or
the number of boosting rounds is hard-coded in the frontend.

## Files (`bundle/`, uploaded to Hugging Face, pinned by `web/model-lock.json`)

| file | content |
|---|---|
| `manifest.json` | schema `writerhand.bundle/2`; `bundle_id`; `files_sha256`; `datasets.<id>.models.<model>` |
| `<dataset>/models/<model>.onnx` | the deployed graph (`pixels` [N,784] in [0,1] -> `probabilities`, plus runtime outputs) |
| `<dataset>/models/<model>.params.bin` | learned parameters, float32 little endian, indexed by `trace.params.tensors` |
| `<dataset>/models/adaboost.bin` | SAMME ensemble (binary, see `adaboost_to_binary`); no ONNX graph |
| `<dataset>/test_vectors.json` | raw test images and expected predictions / probabilities / logits from Python |
| `<dataset>/results.json` | measured metrics shown on the dashboard |

`web/model-lock.json` = `{repo, revision, bundle_id, manifest_sha256}`. The site fetches
`https://huggingface.co/<repo>/resolve/<revision>/...` only; CI fails if the committed
`web/public/data/manifest.json` differs from the lock or from the remote manifest.

## `datasets.<id>`

`title`, `classes` (display names; index = model output index), `preprocessing`
(`mean`, `std`, `eps`, formula), `dataset` (split sizes, seed, split fingerprint), `models`.

## `models.<id>`

`title`, `class`, `format` (`onnx` | `binary`), `file`, `sha256`, `size_bytes`, `params`
(hyper-parameters), `validation` (export vs Python parity), and **`trace`**:

```jsonc
{
  "schema": "writerhand.trace/1",
  "kind": "mlp" | "cnn" | "softmax" | "naive_bayes" | "adaboost",
  "preprocessing": [{"op":"scale","divisor":255}, {"op":"standardize","mean":..,"std":..,"eps":1e-8}],
  "input":  {"name":"pixels","shape":["N",784],"range":[0,1]},
  "layers": [ {"id","op","inputs":[ids],"out_shape":[..],"runtime":"<onnx output>"|null, ...op fields} ],
  "runtime_outputs": [ {"name","layer","shape":[..],"role"} ],   // exactly the extra ONNX outputs
  "output": {"name","kind":"softmax_probability"|"vote_share","calibrated":bool,"classes":[..]},
  "params": {"file","sha256","nbytes","tensors":[{"name","shape","offset","nbytes","layer","role"}]},
  "n_parameters": int
}
```

`layers` is a topologically ordered DAG (every `inputs` id appears earlier). Ops and their fields:

| op | fields | learned tensors (`params`) |
|---|---|---|
| `input`, `flatten`, `reshape` | `out_shape` | – |
| `standardize` | `mean`, `std`, `eps` | – |
| `dense` | `in_features`, `out_features` | `weight` [in,out] (`z = a·W + b`), `bias` |
| `relu`, `softmax` | – | – |
| `conv2d` | `kernel_shape` [Cout,Cin,k,k], `stride`, `padding` | `kernel`, `bias` (cross-correlation, as ONNX `Conv`) |
| `maxpool2d` | `window`, `stride` | – |
| `gaussian_loglik` | `n_classes`, `n_features` | `mean` [K,784], `variance` [K,784], `log_prior` [K] (standardised units) |
| `tree_ensemble` | `n_estimators`, `node_counts[]`, `depths[]`, `split_rule` | in `adaboost.bin` |
| `weighted_vote`, `normalize` | `n_classes` | `alphas` in `adaboost.bin` |

`runtime_outputs[].role`: `standardized_input`, `pre_activation`, `activation`, `pooled`, `logits`,
`log_joint`. For one input the explorer asks ONNX Runtime for exactly these names (plus
`probabilities`), so every displayed intermediate value is produced by the deployed graph.
Parameters are read from `params.bin`; the explorer recomputes selected quantities from them
(weighted sums, convolution patches, log-likelihood terms) and shows the reconstruction next to
the runtime value, with the absolute difference.

## Honesty rules enforced by tests

* `trace.layers` shapes equal the shapes of the ONNX outputs (`tests/test_export.py`).
* `params.bin` tensors are bit-identical to the ONNX initialisers where both exist.
* Reconstructions (MLP weighted sums, convolution/pooling values, GaussianNB log-likelihood
  decomposition, softmax logits, AdaBoost paths and votes) agree with the runtime outputs.
* AdaBoost `vote_share` is **not calibrated** (`output.calibrated = false`).
* An activation map shows a layer's activation for this input; it is not an attribution or a causal
  explanation of the decision. Methods that do attribute (e.g. pixel-wise `x_i·W_ik` for the linear
  models, which is exact for them) are labelled as such.
