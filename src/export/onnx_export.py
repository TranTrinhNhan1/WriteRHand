"""Build ONNX graphs by hand from fitted NumPy / torch parameters.

Every graph has the same contract:
    input   ``pixels``         float32 [N, 784]  raw pixels scaled to [0, 1] (NOT standardised)
    output  ``probabilities``  float32 [N, 10]
The preprocessing ``(x - mean) / (std + 1e-8)`` is the first node of the graph, so a browser only
has to hand over pixel / 255 and the training-time normalisation is applied identically.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17
IR_VERSION = 8  # widely supported by onnxruntime / onnxruntime-web


def _init(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)


def _wrap(nodes, inits, doc, n_classes, class_names):
    graph = helper.make_graph(
        nodes,
        "writerhand",
        [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, ["batch", 784])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", n_classes])],
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
    return _wrap(
        nodes,
        inits,
        "MLP (ReLU) trained with hand-written backprop",
        len(model.classes),
        model.classes,
    )


def softmax_regression_to_onnx(model, mean, std):
    nodes, inits = _normalise_nodes(mean, std)
    inits += [_init("W", model.W), _init("b", model.b)]
    nodes += [
        helper.make_node("Gemm", ["x", "W", "b"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    return _wrap(nodes, inits, "Multiclass softmax regression", len(model.classes), model.classes)


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
    return _wrap(
        nodes,
        inits,
        "Gaussian Naive Bayes (log joint likelihood -> posterior)",
        len(model.classes),
        model.classes,
    )


def adaboost_to_json(model, mean, std):
    """Compact JSON of the SAMME ensemble (evaluated by a tiny JS function in the browser)."""
    trees = []
    for t in range(model.n_estimators_):
        n = (
            int(np.flatnonzero(model.tree_feature[t] >= 0).max() + 1)
            if (model.tree_feature[t] >= 0).any()
            else 0
        )
        n = max(n, int(max(model.tree_left[t].max(), model.tree_right[t].max())) + 1, 1)
        trees.append(
            {
                "f": model.tree_feature[t][:n].tolist(),
                "t": [float(x) for x in model.tree_threshold[t][:n]],
                "l": model.tree_left[t][:n].tolist(),
                "r": model.tree_right[t][:n].tolist(),
                "c": model.tree_leaf[t][:n].tolist(),
            }
        )
    return {
        "type": "adaboost-samme",
        "classes": [int(c) for c in model.classes],
        "alphas": [float(a) for a in model.alphas],
        "trees": trees,
        "preprocess": {"mean": mean, "std": std, "eps": 1e-8},
        "note": "x = (pixel/255 - mean)/(std+eps); go left when x[f] < threshold",
    }


def adaboost_predict_json(spec, pixels01):
    """Reference evaluator of the exported JSON (mirrors the JS code; used in tests)."""
    x = (np.asarray(pixels01, dtype=np.float64) - spec["preprocess"]["mean"]) / (
        spec["preprocess"]["std"] + spec["preprocess"]["eps"]
    )
    scores = np.zeros((len(x), len(spec["classes"])))
    for tree, a in zip(spec["trees"], spec["alphas"], strict=True):
        for i, row in enumerate(x):
            node = 0
            while tree["f"][node] >= 0:
                node = (
                    tree["l"][node] if row[tree["f"][node]] < tree["t"][node] else tree["r"][node]
                )
            scores[i, tree["c"][node]] += a
    return scores / scores.sum(1, keepdims=True)
