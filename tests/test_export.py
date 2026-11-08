import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import numpy_helper  # noqa: E402

from src.export.onnx_export import (  # noqa: E402
    adaboost_from_binary,
    adaboost_predict_binary,
    adaboost_to_binary,
    cnn_to_onnx,
    gaussian_nb_to_onnx,
    mlp_to_onnx,
    softmax_regression_to_onnx,
)
from src.export.tracespec import build_trace  # noqa: E402
from src.models.adaboost import AdaBoostSAMME  # noqa: E402
from src.models.cnn import CNN  # noqa: E402
from src.models.logistic_regression import SoftmaxRegression  # noqa: E402
from src.models.mlp import MLP  # noqa: E402
from src.models.naive_bayes import GaussianNB  # noqa: E402

MEAN, STD = 0.3, 0.4
NAMES = [f"c{i}" for i in range(10)]


@pytest.fixture(scope="module")
def pixel_data():
    """Pixels in [0,1]; the models see (p - MEAN) / STD like in the real pipeline."""
    rng = np.random.default_rng(0)
    centers = rng.random((10, 784)) > 0.8
    y = np.repeat(np.arange(10), 40)
    P = np.clip(centers[y] * 0.9 + rng.normal(0, 0.1, (400, 784)), 0, 1).astype(np.float32)
    return P, y


def _run_all(proto, P):
    sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, {"pixels": P.astype(np.float32)}), strict=True))


def _norm(P):
    return (P.astype(np.float64) - MEAN) / (STD + 1e-8)


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _unpack(trace, pack):
    blob, out = pack.blob(), {}
    for t in pack.entries:
        n = int(np.prod(t["shape"]))
        out[t["name"]] = np.frombuffer(blob, "<f4", n, t["offset"]).reshape(t["shape"])
    return out


@pytest.fixture(scope="module")
def models(pixel_data):
    P, y = pixel_data
    X = _norm(P)
    return {
        "mlp": MLP((32, 16), epochs=3, learning_rate=0.05).fit(X, y),
        "softmax": SoftmaxRegression(epochs=3).fit(X, y),
        "naive_bayes": GaussianNB(0.5).fit(X, y),
        "cnn": CNN((3, 4), dense=(12,), epochs=2, batch_size=64).fit(X, y),
    }


BUILDERS = {
    "mlp": mlp_to_onnx,
    "softmax": softmax_regression_to_onnx,
    "naive_bayes": gaussian_nb_to_onnx,
    "cnn": cnn_to_onnx,
}
TOL = {"mlp": 1e-5, "softmax": 1e-5, "naive_bayes": 1e-3, "cnn": 1e-4}


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_onnx_matches_python_model(kind, models, pixel_data):
    P, _ = pixel_data
    model = models[kind]
    out = _run_all(BUILDERS[kind](model, MEAN, STD), P)["probabilities"]
    ref = model.predict_proba(_norm(P))
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < TOL[kind]
    np.testing.assert_array_equal(out.argmax(1), ref.argmax(1))
    np.testing.assert_allclose(out.sum(1), 1.0, atol=1e-5)


def test_onnx_supports_single_sample_and_batches(models, pixel_data):
    P, _ = pixel_data
    proto = mlp_to_onnx(models["mlp"], MEAN, STD)
    one, many = _run_all(proto, P[:1])["probabilities"], _run_all(proto, P[:7])["probabilities"]
    np.testing.assert_allclose(one[0], many[0], atol=1e-6)


# ------------------------------------------------------------------ trace contract
@pytest.mark.parametrize("kind", list(BUILDERS))
def test_manifest_runtime_outputs_match_the_graph(kind, models, pixel_data):
    """Declared runtime outputs == the extra outputs and shapes of the ONNX graph."""
    P, _ = pixel_data
    model = models[kind]
    trace, _ = build_trace(kind, model, MEAN, STD, NAMES)
    outs = _run_all(BUILDERS[kind](model, MEAN, STD), P[:3])
    declared = {r["name"]: r["shape"] for r in trace["runtime_outputs"]}
    assert set(outs) - {"probabilities"} == set(declared)
    for name, shape in declared.items():
        assert list(outs[name].shape[1:]) == shape, name


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_layer_graph_is_consistent(kind, models):
    trace, _ = build_trace(kind, models[kind], MEAN, STD, NAMES)
    ids = [layer["id"] for layer in trace["layers"]]
    assert len(set(ids)) == len(ids)
    for layer in trace["layers"]:
        assert all(i in ids[: ids.index(layer["id"])] for i in layer["inputs"])  # DAG, topological
    assert trace["layers"][-1]["op"] == "softmax" and trace["output"]["calibrated"] is True
    assert trace["schema"] == "writerhand.trace/1"


def test_architecture_metadata_equals_checkpoint_topology(models):
    mlp = models["mlp"]
    trace, pack = build_trace("mlp", mlp, MEAN, STD, NAMES)
    dense = [layer for layer in trace["layers"] if layer["op"] == "dense"]
    assert [d["out_features"] for d in dense] == [*mlp.hidden_sizes, 10]
    assert [d["in_features"] for d in dense] == [784, *mlp.hidden_sizes]
    assert [t["shape"] for t in pack.entries if t["role"] == "weight"] == [
        [d["in_features"], d["out_features"]] for d in dense
    ]
    cnn = models["cnn"]
    trace, pack = build_trace("cnn", cnn, MEAN, STD, NAMES)
    by_id = {layer["id"]: layer for layer in trace["layers"]}
    assert by_id["conv0"]["kernel_shape"] == [3, 1, 3, 3] and by_id["conv1"]["kernel_shape"] == [
        4,
        3,
        3,
        3,
    ]
    assert by_id["conv0"]["out_shape"] == [3, 26, 26] and by_id["pool0"]["out_shape"] == [3, 13, 13]
    assert by_id["pool1"]["out_shape"] == [4, 5, 5] and by_id["flat"]["out_shape"] == [100]
    assert by_id["z0"]["in_features"] == 100 and by_id["z0"]["out_features"] == 12
    assert trace["n_parameters"] == cnn.n_parameters()


@pytest.mark.parametrize("kind", ["mlp", "softmax", "naive_bayes", "cnn"])
def test_params_bin_equals_onnx_initializers(kind, models):
    model = models[kind]
    trace, pack = build_trace(kind, model, MEAN, STD, NAMES)
    inits = {
        i.name: numpy_helper.to_array(i) for i in BUILDERS[kind](model, MEAN, STD).graph.initializer
    }
    got = _unpack(trace, pack)
    for name, arr in got.items():
        if name in inits:  # NB class_mean/var/log_prior are stored in a different (quadratic) form
            np.testing.assert_array_equal(arr, inits[name])
    assert got  # every model ships its learned parameters


def test_mlp_weighted_sums_and_activations_reconstruct(models, pixel_data):
    P, _ = pixel_data
    mlp = models["mlp"]
    trace, pack = build_trace("mlp", mlp, MEAN, STD, NAMES)
    prm, o = _unpack(trace, pack), _run_all(mlp_to_onnx(mlp, MEAN, STD), P[:4])
    a = o["x"].astype(np.float64)
    for i in range(2):
        z = a @ prm[f"W{i}"] + prm[f"b{i}"]
        np.testing.assert_allclose(z, o[f"z{i}"], atol=1e-4)
        a = np.maximum(z, 0)
        np.testing.assert_allclose(a, o[f"a{i}"], atol=1e-4)
    np.testing.assert_allclose(a @ prm["W2"] + prm["b2"], o["logits"], atol=1e-4)
    # one neuron, term by term (what the explorer shows when a neuron is selected)
    j = 5
    terms = o["x"][0].astype(np.float64) * prm["W0"][:, j]
    assert terms.sum() + prm["b0"][j] == pytest.approx(o["z0"][0, j], abs=1e-4)


def test_softmax_logit_reconstruction(models, pixel_data):
    P, _ = pixel_data
    trace, pack = build_trace("softmax", models["softmax"], MEAN, STD, NAMES)
    prm, o = (
        _unpack(trace, pack),
        _run_all(softmax_regression_to_onnx(models["softmax"], MEAN, STD), P[:4]),
    )
    contrib = o["x"][:, :, None] * prm["W"][None]  # (N, pixel, class)
    np.testing.assert_allclose(contrib.sum(1) + prm["b"], o["logits"], atol=1e-3)
    np.testing.assert_allclose(_softmax(o["logits"]), o["probabilities"], atol=1e-5)


def test_naive_bayes_likelihood_decomposition(models, pixel_data):
    P, _ = pixel_data
    nb = models["naive_bayes"]
    trace, pack = build_trace("naive_bayes", nb, MEAN, STD, NAMES)
    prm, o = _unpack(trace, pack), _run_all(gaussian_nb_to_onnx(nb, MEAN, STD), P[:4])
    x = o["x"].astype(np.float64)
    var, mu = prm["class_var"].astype(np.float64), prm["class_mean"].astype(np.float64)
    ll = -0.5 * (np.log(2 * np.pi * var)[None] + (x[:, None, :] - mu[None]) ** 2 / var[None])
    logits = ll.sum(2) + prm["log_prior"]
    np.testing.assert_allclose(logits, o["logits"], atol=2e-2, rtol=1e-4)
    # and it is the Python model's log joint likelihood
    np.testing.assert_allclose(logits, nb._joint_log_likelihood(_norm(P[:4])), atol=2e-2, rtol=1e-4)


def test_cnn_convolution_and_pooling_values(models, pixel_data):
    P, _ = pixel_data
    cnn = models["cnn"]
    trace, pack = build_trace("cnn", cnn, MEAN, STD, NAMES)
    prm, o = _unpack(trace, pack), _run_all(cnn_to_onnx(cnn, MEAN, STD), P[:3])
    ref = cnn.forward_trace(_norm(P[:3]))
    for key in ("conv0", "act0", "pool0", "conv1", "act1", "pool1", "z0", "a0", "logits"):
        np.testing.assert_allclose(o[key], ref[key], atol=2e-4, err_msg=key)
    # one output pixel of a feature map, multiplied and summed by hand
    K, kb = prm["K1"], prm["kb1"]
    c, r, q = 2, 4, 6
    patch = o["pool0"][0][:, r : r + 3, q : q + 3]
    assert float((patch * K[c]).sum() + kb[c]) == pytest.approx(o["conv1"][0, c, r, q], abs=1e-4)
    # max pool picks the window maximum of the activation map
    win = o["act1"][0, 1, 2:4, 4:6]
    assert win.max() == pytest.approx(o["pool1"][0, 1, 1, 2])


# ------------------------------------------------------------------ AdaBoost
def test_adaboost_binary_roundtrip_matches_python(pixel_data):
    P, y = pixel_data
    X = _norm(P)
    m = AdaBoostSAMME(n_estimators=8, max_depth=3).fit(X, y)
    blob = adaboost_to_binary(m, MEAN, STD)
    spec = adaboost_from_binary(blob)
    assert len(blob) % 8 == 0 and spec["classes"] == list(range(10))
    out = adaboost_predict_binary(spec, P)
    np.testing.assert_allclose(out, m.predict_proba(X), atol=1e-12)
    assert np.array_equal(np.array(spec["classes"])[out.argmax(1)], m.predict(X))
    assert len(spec["node_counts"]) == len(spec["depths"]) == len(spec["alphas"]) == 8
    trace, pack = build_trace(
        "adaboost",
        m,
        MEAN,
        STD,
        NAMES,
        {"n_estimators": 8, "node_counts": spec["node_counts"], "depths": spec["depths"]},
    )
    assert pack.entries == [] and trace["output"]["calibrated"] is False
    assert trace["output"]["kind"] == "vote_share"
