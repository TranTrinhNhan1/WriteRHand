"""Trace contract between the Python models and the React "Model Execution Explorer".

Everything the explorer draws is described here, *generated from the fitted model object*: layer
list and connections, tensor shapes, preprocessing constants, the learned parameters
(``<model>.params.bin``) and the names of the per-sample tensors the ONNX graph returns. Nothing
about depth, widths, kernel sizes or the number of boosting rounds is hard-coded in the frontend.
The specification in prose is ``docs/TRACE_CONTRACT.md``; the frontend validates every manifest
against it before drawing anything.
"""

import hashlib

import numpy as np

TRACE_SCHEMA = "writerhand.trace/1"
BUNDLE_SCHEMA = "writerhand.bundle/2"
EPS = 1e-8


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ParamPack:
    """Collects named float32 tensors into one little-endian blob with an index."""

    def __init__(self):
        self.entries, self._chunks, self._pos = [], [], 0

    def add(self, name, arr, layer, role):
        a = np.ascontiguousarray(arr, dtype="<f4")
        self.entries.append(
            {
                "name": name,
                "shape": list(a.shape),
                "dtype": "float32",
                "offset": self._pos,
                "nbytes": int(a.nbytes),
                "layer": layer,
                "role": role,
            }
        )
        self._chunks.append(a.tobytes())
        self._pos += a.nbytes

    def blob(self):
        return b"".join(self._chunks)


def _preprocessing(mean, std):
    return [
        {"op": "scale", "divisor": 255.0, "note": "uint8 pixel -> [0, 1]"},
        {
            "op": "standardize",
            "mean": mean,
            "std": std,
            "eps": EPS,
            "formula": "(p - mean)/(std + eps)",
        },
    ]


def _front(mean, std):
    return [
        {"id": "image", "op": "input", "label": "image", "out_shape": [28, 28], "inputs": []},
        {"id": "flatten", "op": "flatten", "out_shape": [784], "inputs": ["image"]},
        {
            "id": "standardize",
            "op": "standardize",
            "mean": mean,
            "std": std,
            "eps": EPS,
            "out_shape": [784],
            "inputs": ["flatten"],
            "runtime": "x",
        },
    ]


def _dense_head(pack, layers, prev, W, b, ids, relu_ids, name_w, name_b, last):
    """Append dense (+ReLU) layer(s); returns the id of the last layer appended."""
    zid, aid = ids
    pack.add(name_w, W, zid, "weight")
    pack.add(name_b, b, zid, "bias")
    layers.append(
        {
            "id": zid,
            "op": "dense",
            "in_features": int(W.shape[0]),
            "out_features": int(W.shape[1]),
            "params": {"weight": name_w, "bias": name_b},
            "out_shape": [int(W.shape[1])],
            "inputs": [prev],
            "runtime": zid,
        }
    )
    if last:
        return zid
    layers.append(
        {"id": aid, "op": "relu", "out_shape": [int(W.shape[1])], "inputs": [zid], "runtime": aid}
    )
    return aid


def build_trace(kind, model, mean, std, class_names, extra=None):
    """Return ``(trace, ParamPack)`` for a fitted model.

    ``kind`` is one of ``mlp | cnn | softmax | svm | naive_bayes | adaboost``. The trace dictionary goes
    into ``manifest.json``; the pack is written to ``<model>.params.bin`` by the exporter.
    """
    pack = ParamPack()
    layers = _front(mean, std)
    runtime = [{"name": "x", "layer": "standardize", "shape": [784], "role": "standardized_input"}]
    k = len(class_names)

    if kind == "mlp":
        prev = "standardize"
        ws = model.weights_numpy()
        for i, (W, b) in enumerate(ws):
            last = i == len(ws) - 1
            zid = "logits" if last else f"z{i}"
            prev = _dense_head(
                pack, layers, prev, W, b, (zid, f"a{i}"), None, f"W{i}", f"b{i}", last
            )
            if not last:
                runtime += [
                    {
                        "name": f"z{i}",
                        "layer": zid,
                        "shape": [int(W.shape[1])],
                        "role": "pre_activation",
                    },
                    {
                        "name": f"a{i}",
                        "layer": f"a{i}",
                        "shape": [int(W.shape[1])],
                        "role": "activation",
                    },
                ]
        runtime.append({"name": "logits", "layer": "logits", "shape": [k], "role": "logits"})
        out_kind = "softmax_probability"
    elif kind in ("softmax", "svm"):
        W, b = model.W, model.b
        _dense_head(pack, layers, "standardize", W, b, ("logits", "a"), None, "W", "b", True)
        runtime.append({"name": "logits", "layer": "logits", "shape": [k], "role": "logits"})
        prev = "logits"
        # SVM scores are not probabilities: the softmax only normalises them (never calibrated)
        out_kind = "softmax_probability" if kind == "softmax" else "softmax_score"
    elif kind == "naive_bayes":
        var, mu = model.var, model.mean
        pack.add("class_mean", mu, "logits", "class_mean")
        pack.add("class_var", var, "logits", "class_variance")
        pack.add("log_prior", np.log(model.prior), "logits", "log_prior")
        layers.append(
            {
                "id": "logits",
                "op": "gaussian_loglik",
                "params": {"mean": "class_mean", "variance": "class_var", "log_prior": "log_prior"},
                "n_classes": k,
                "n_features": 784,
                "out_shape": [k],
                "inputs": ["standardize"],
                "runtime": "logits",
            }
        )
        runtime.append({"name": "logits", "layer": "logits", "shape": [k], "role": "log_joint"})
        prev = "logits"
        out_kind = "softmax_probability"
    elif kind == "cnn":
        w = model.weights_numpy()
        desc = {layer["id"]: layer for layer in model.describe()}
        layers.append(
            {
                "id": "reshape",
                "op": "reshape",
                "out_shape": [1, 28, 28],
                "inputs": ["standardize"],
            }
        )
        prev = "reshape"
        for s, (K, kb) in enumerate(zip(w["K"], w["kb"], strict=True)):
            pack.add(f"K{s}", K, f"conv{s}", "kernel")
            pack.add(f"kb{s}", kb, f"conv{s}", "bias")
            layers += [
                {
                    "id": f"conv{s}",
                    "op": "conv2d",
                    "kernel_shape": list(K.shape),
                    "stride": 1,
                    "padding": 0,
                    "params": {"kernel": f"K{s}", "bias": f"kb{s}"},
                    "out_shape": desc[f"conv{s}"]["out_shape"],
                    "inputs": [prev],
                    "runtime": f"conv{s}",
                },
                {
                    "id": f"act{s}",
                    "op": "relu",
                    "out_shape": desc[f"act{s}"]["out_shape"],
                    "inputs": [f"conv{s}"],
                    "runtime": f"act{s}",
                },
                {
                    "id": f"pool{s}",
                    "op": "maxpool2d",
                    "window": model.pool,
                    "stride": model.pool,
                    "out_shape": desc[f"pool{s}"]["out_shape"],
                    "inputs": [f"act{s}"],
                    "runtime": f"pool{s}",
                },
            ]
            for pre, role in (
                ("conv", "pre_activation"),
                ("act", "activation"),
                ("pool", "pooled"),
            ):
                runtime.append(
                    {
                        "name": f"{pre}{s}",
                        "layer": f"{pre}{s}",
                        "shape": desc[f"{pre}{s}"]["out_shape"],
                        "role": role,
                    }
                )
            prev = f"pool{s}"
        layers.append(
            {
                "id": "flat",
                "op": "flatten",
                "out_shape": desc["flat"]["out_shape"],
                "inputs": [prev],
            }
        )
        prev = "flat"
        ws = list(zip(w["W"], w["b"], strict=True))
        for i, (W, b) in enumerate(ws):
            last = i == len(ws) - 1
            zid = "logits" if last else f"z{i}"
            prev = _dense_head(
                pack, layers, prev, W, b, (zid, f"a{i}"), None, f"W{i}", f"b{i}", last
            )
            if not last:
                runtime += [
                    {
                        "name": f"z{i}",
                        "layer": zid,
                        "shape": [int(W.shape[1])],
                        "role": "pre_activation",
                    },
                    {
                        "name": f"a{i}",
                        "layer": f"a{i}",
                        "shape": [int(W.shape[1])],
                        "role": "activation",
                    },
                ]
        runtime.append({"name": "logits", "layer": "logits", "shape": [k], "role": "logits"})
        prev = "logits"
        out_kind = "softmax_probability"
    elif kind == "adaboost":
        extra = extra or {}
        layers.append(
            {
                "id": "trees",
                "op": "tree_ensemble",
                "n_estimators": int(extra["n_estimators"]),
                "node_counts": extra["node_counts"],
                "depths": extra["depths"],
                "split_rule": "go left when x[feature] < threshold",
                "out_shape": [int(extra["n_estimators"])],
                "inputs": ["standardize"],
            }
        )
        layers.append(
            {
                "id": "vote",
                "op": "weighted_vote",
                "n_classes": k,
                "out_shape": [k],
                "inputs": ["trees"],
                "note": "score_k = sum over rounds of alpha_t * 1[tree_t predicts k]",
            }
        )
        prev = "vote"
        runtime = []  # no ONNX graph: the ensemble file itself is evaluated in the browser
        out_kind = "vote_share"
    else:
        raise KeyError(kind)

    final = {
        "id": "output",
        "op": "softmax" if out_kind in ("softmax_probability", "softmax_score") else "normalize",
        "out_shape": [k],
        "inputs": [prev],
        "runtime": "probabilities" if out_kind != "vote_share" else None,
    }
    layers.append(final)
    trace = {
        "schema": TRACE_SCHEMA,
        "kind": kind,
        "preprocessing": _preprocessing(mean, std),
        "input": {
            "name": "pixels",
            "shape": ["N", 784],
            "range": [0, 1],
            "layout": "28x28 row-major",
        },
        "layers": layers,
        "runtime_outputs": runtime,
        "output": {
            "name": "probabilities" if out_kind != "vote_share" else "vote_share",
            "kind": out_kind,
            "calibrated": out_kind == "softmax_probability",
            "classes": list(class_names),
        },
        "n_parameters": int(sum(int(np.prod(e["shape"])) for e in pack.entries)) or None,
    }
    return trace, pack
