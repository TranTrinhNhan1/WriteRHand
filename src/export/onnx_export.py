"""Build ONNX graphs by hand from fitted NumPy / torch parameters.

Every graph has the same contract:
    input   ``pixels``         float32 [N, 784]  raw pixels scaled to [0, 1] (NOT standardised)
    output  ``probabilities``  float32 [N, K]
Extra, optional outputs ("runtime trace") expose the per-sample intermediate tensors for the model
execution explorer; they never change the probabilities. They are listed with their shapes in the
``trace`` section of the manifest (see ``docs/TRACE_CONTRACT.md``). The learned parameters are
shipped once in a separate ``<model>.params.bin`` (same float32 values as the graph initializers).
The preprocessing ``(x - mean) / (std + 1e-8)`` is the first node of the graph, so a browser only
has to hand over pixel / 255 and the training-time normalisation is applied identically.
"""

import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17
IR_VERSION = 8  # widely supported by onnxruntime / onnxruntime-web


def _init(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)


def _wrap(nodes, inits, doc, n_classes, class_names, trace=()):
    """``trace``: (output name, shape) pairs for the extra, optional outputs."""
    outputs = [
        helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", n_classes])
    ]
    outputs += [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(sh)) for n, sh in trace]
    graph = helper.make_graph(
        nodes,
        "writerhand",
        [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, ["batch", 784])],
        outputs,
        initializer=inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="writerhand",
        doc_string=doc,
    )
    model.ir_version = IR_VERSION
    meta = {"classes": ",".join(str(c) for c in class_names)}
    for k, v in meta.items():
        p = model.metadata_props.add()
        p.key, p.value = k, v
    onnx.checker.check_model(model)
    return model


def _normalise_nodes(mean, std):
    inits = [_init("mean", [mean]), _init("denom", [std + 1e-8])]
    nodes = [
        helper.make_node("Sub", ["pixels", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "denom"], ["x"]),
    ]
    return nodes, inits


def mlp_to_onnx(model, mean, std):
    nodes, inits = _normalise_nodes(mean, std)
    a = "x"
    layers = model.weights_numpy()
    for i, (W, b) in enumerate(layers):
        inits += [_init(f"W{i}", W), _init(f"b{i}", b)]
        z = "logits" if i == len(layers) - 1 else f"z{i}"
        nodes.append(helper.make_node("Gemm", [a, f"W{i}", f"b{i}"], [z]))
        if i < len(layers) - 1:
            nodes.append(helper.make_node("Relu", [z], [f"a{i}"]))
            a = f"a{i}"
    nodes.append(helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1))
    trace = [("x", ["batch", 784])]
    for i, (W, _) in enumerate(layers[:-1]):
        trace += [(f"z{i}", ["batch", W.shape[1]]), (f"a{i}", ["batch", W.shape[1]])]
    trace.append(("logits", ["batch", len(model.classes)]))
    return _wrap(
        nodes,
        inits,
        "MLP (ReLU) trained with hand-written backprop",
        len(model.classes),
        model.classes,
        trace,
    )


def softmax_regression_to_onnx(model, mean, std):
    nodes, inits = _normalise_nodes(mean, std)
    inits += [_init("W", model.W), _init("b", model.b)]
    nodes += [
        helper.make_node("Gemm", ["x", "W", "b"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    k = len(model.classes)
    trace = [("x", ["batch", 784]), ("logits", ["batch", k])]
    return _wrap(
        nodes, inits, "Multiclass softmax regression", len(model.classes), model.classes, trace
    )


def gaussian_nb_to_onnx(model, mean, std):
    """log p(k) - 0.5 sum_j [log(2 pi s2_kj) + (x_j - m_kj)^2 / s2_kj], expanded as a quadratic
    form so it becomes two matrix products:  (x^2) A + x B + c."""
    var, mu = model.var, model.mean  # (K, D)
    A = (-0.5 / var).T  # (D, K)
    B = (mu / var).T
    c = np.log(model.prior) - 0.5 * np.log(2 * np.pi * var).sum(1) - 0.5 * (mu**2 / var).sum(1)
    nodes, inits = _normalise_nodes(mean, std)
    inits += [_init("A", A), _init("B", B), _init("c", c)]
    nodes += [
        helper.make_node("Mul", ["x", "x"], ["x2"]),
        helper.make_node("MatMul", ["x2", "A"], ["t1"]),
        helper.make_node("MatMul", ["x", "B"], ["t2"]),
        helper.make_node("Add", ["t1", "t2"], ["t3"]),
        helper.make_node("Add", ["t3", "c"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    k = len(model.classes)
    trace = [("x", ["batch", 784]), ("logits", ["batch", k])]
    return _wrap(
        nodes,
        inits,
        "Gaussian Naive Bayes (log joint likelihood -> posterior)",
        len(model.classes),
        model.classes,
        trace,
    )


def cnn_to_onnx(model, mean, std):
    """Conv -> Relu -> MaxPool stages, Flatten, dense head. ONNX ``Conv`` is a cross-correlation,
    exactly what ``src/models/cnn.py`` computes."""
    nodes, inits = _normalise_nodes(mean, std)
    inits.append(numpy_helper.from_array(np.array([-1, 1, 28, 28], dtype=np.int64), "img_shape"))
    nodes.append(helper.make_node("Reshape", ["x", "img_shape"], ["x_img"]))
    w = model.weights_numpy()
    a = "x_img"
    trace = [("x", ["batch", 784])]
    h = 28
    for s_, (K, kb) in enumerate(zip(w["K"], w["kb"], strict=True)):
        inits += [_init(f"K{s_}", K), _init(f"kb{s_}", kb)]
        nodes.append(
            helper.make_node(
                "Conv", [a, f"K{s_}", f"kb{s_}"], [f"conv{s_}"], pads=[0, 0, 0, 0], strides=[1, 1]
            )
        )
        nodes.append(helper.make_node("Relu", [f"conv{s_}"], [f"act{s_}"]))
        nodes.append(
            helper.make_node(
                "MaxPool",
                [f"act{s_}"],
                [f"pool{s_}"],
                kernel_shape=[model.pool, model.pool],
                strides=[model.pool, model.pool],
            )
        )
        hconv = h - K.shape[2] + 1
        h = hconv // model.pool
        c = K.shape[0]
        a = f"pool{s_}"
        trace += [
            (f"conv{s_}", ["batch", c, hconv, hconv]),
            (f"act{s_}", ["batch", c, hconv, hconv]),
            (f"pool{s_}", ["batch", c, h, h]),
        ]
    nodes.append(helper.make_node("Flatten", [a], ["flat"], axis=1))
    a = "flat"
    n_dense = len(w["W"])
    for i, (W, b) in enumerate(zip(w["W"], w["b"], strict=True)):
        inits += [_init(f"W{i}", W), _init(f"b{i}", b)]
        z = "logits" if i == n_dense - 1 else f"z{i}"
        nodes.append(helper.make_node("Gemm", [a, f"W{i}", f"b{i}"], [z]))
        if i < n_dense - 1:
            nodes.append(helper.make_node("Relu", [z], [f"a{i}"]))
            trace += [(f"z{i}", ["batch", W.shape[1]]), (f"a{i}", ["batch", W.shape[1]])]
            a = f"a{i}"
    trace.append(("logits", ["batch", len(model.classes)]))
    nodes.append(helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1))
    return _wrap(
        nodes,
        inits,
        "Small CNN (Conv-ReLU-MaxPool stages + dense head) trained with hand-written backprop",
        len(model.classes),
        model.classes,
        trace,
    )


# --------------------------------------------------------------------------- AdaBoost (binary)
ADA_MAGIC = b"WRHADA01"


def adaboost_to_binary(model, mean, std):
    """Compact binary for the SAMME ensemble (about half the size of the former JSON and parsed
    with typed-array views, no ``JSON.parse`` of tens of MB).

    Layout (little endian)::

        8 B   magic ``WRHADA01``
        4 B   uint32 header length H            (header is padded with spaces to a multiple of 8)
        H B   UTF-8 JSON header: classes, alphas, preprocess, node_counts, max_depth
        then, concatenated over trees in order and each padded to 8 bytes:
        float64 threshold[N] ; int16 feature[N] ; int16 left[N] ; int16 right[N] ; uint8 klass[N]
    """
    n_nodes, parts = [], []
    for t in range(model.n_estimators_):
        n = (
            int(np.flatnonzero(model.tree_feature[t] >= 0).max() + 1)
            if (model.tree_feature[t] >= 0).any()
            else 0
        )
        n = max(n, int(max(model.tree_left[t].max(), model.tree_right[t].max())) + 1, 1)
        n_nodes.append(n)
        parts.append(
            (
                np.asarray(model.tree_threshold[t][:n], dtype="<f8"),
                np.asarray(model.tree_feature[t][:n], dtype="<i2"),
                np.asarray(model.tree_left[t][:n], dtype="<i2"),
                np.asarray(model.tree_right[t][:n], dtype="<i2"),
                np.asarray(model.tree_leaf[t][:n], dtype="u1"),
            )
        )
    depths = []
    for t in range(model.n_estimators_):
        f, l_, r_ = parts[t][1], parts[t][2], parts[t][3]
        depth, stack = 0, [(0, 0)]
        while stack:
            node, d = stack.pop()
            depth = max(depth, d)
            if f[node] >= 0:
                stack += [(int(l_[node]), d + 1), (int(r_[node]), d + 1)]
        depths.append(depth)
    header = {
        "type": "adaboost-samme",
        "version": 1,
        "classes": [int(c) for c in model.classes],
        "alphas": [float(a) for a in model.alphas],
        "preprocess": {"mean": mean, "std": std, "eps": 1e-8},
        "node_counts": n_nodes,
        "depths": depths,
        "rule": "x = (pixel/255 - mean)/(std+eps); go left when x[feature] < threshold; leaf: feature = -1",
    }
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * (-(12 + len(hb)) % 8)
    out = [ADA_MAGIC, np.uint32(len(hb)).tobytes(), hb]
    for thr, f, l_, r_, c in parts:
        for arr in (thr, f, l_, r_, c):
            raw = arr.tobytes()
            out.append(raw + b"\0" * (-len(raw) % 8))
    return b"".join(out)


def adaboost_from_binary(buf):
    """Inverse of ``adaboost_to_binary`` (reference reader, mirrors ``web/src/lib/adaboost.ts``)."""
    buf = bytes(buf)
    if buf[:8] != ADA_MAGIC:
        raise ValueError("not a WriteRHand AdaBoost file")
    hl = int(np.frombuffer(buf[8:12], "<u4")[0])
    spec = json.loads(buf[12 : 12 + hl].decode("utf-8"))
    pos = 12 + hl
    trees = []
    for n in spec["node_counts"]:
        tree = {}
        for key, dt, size in (
            ("t", "<f8", 8),
            ("f", "<i2", 2),
            ("l", "<i2", 2),
            ("r", "<i2", 2),
            ("c", "u1", 1),
        ):
            nbytes = n * size
            tree[key] = np.frombuffer(buf, dt, n, pos).astype(
                np.float64 if key == "t" else np.int64
            )
            pos += nbytes + (-nbytes % 8)
        trees.append(tree)
    spec["trees"] = trees
    return spec


def adaboost_scores(spec, pixels01):
    """Vote scores of the binary-format ensemble (float64, like the Python model)."""
    x = (np.asarray(pixels01, dtype=np.float64) - spec["preprocess"]["mean"]) / (
        spec["preprocess"]["std"] + spec["preprocess"]["eps"]
    )
    scores = np.zeros((len(x), len(spec["classes"])))
    for tree, a in zip(spec["trees"], spec["alphas"], strict=True):
        node = np.zeros(len(x), dtype=np.int64)
        active = tree["f"][node] >= 0
        while active.any():
            idx = np.flatnonzero(active)
            f = tree["f"][node[idx]]
            go_left = x[idx, f] < tree["t"][node[idx]]
            node[idx] = np.where(go_left, tree["l"][node[idx]], tree["r"][node[idx]])
            active = tree["f"][node] >= 0
        scores[np.arange(len(x)), tree["c"][node]] += a
    return scores


def adaboost_predict_binary(spec, pixels01):
    s = adaboost_scores(spec, pixels01)
    return s / s.sum(1, keepdims=True)
