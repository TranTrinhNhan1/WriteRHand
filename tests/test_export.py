import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from src.export.onnx_export import (  # noqa: E402
    adaboost_predict_json,
    adaboost_to_json,
    gaussian_nb_to_onnx,
    mlp_to_onnx,
    softmax_regression_to_onnx,
)
from src.models.adaboost import AdaBoostSAMME  # noqa: E402
from src.models.logistic_regression import SoftmaxRegression  # noqa: E402
from src.models.mlp import MLP  # noqa: E402
from src.models.naive_bayes import GaussianNB  # noqa: E402

MEAN, STD = 0.3, 0.4


@pytest.fixture(scope="module")
def pixel_data():
    """Pixels in [0,1]; the models see (p - MEAN) / STD like in the real pipeline."""
    rng = np.random.default_rng(0)
    centers = rng.random((10, 784)) > 0.8
    y = np.repeat(np.arange(10), 40)
    P = np.clip(centers[y] * 0.9 + rng.normal(0, 0.1, (400, 784)), 0, 1).astype(np.float32)
    return P, y


def _run(model_proto, P):
    sess = ort.InferenceSession(model_proto.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(["probabilities"], {"pixels": P.astype(np.float32)})[0]


def _norm(P):
    return (P.astype(np.float64) - MEAN) / (STD + 1e-8)


@pytest.mark.parametrize(
    "name,model,builder,tol",
    [
        ("mlp", MLP((32, 16), epochs=3, learning_rate=0.05), mlp_to_onnx, 1e-5),
        ("softmax", SoftmaxRegression(epochs=3), softmax_regression_to_onnx, 1e-5),
        ("nb", GaussianNB(0.5), gaussian_nb_to_onnx, 1e-3),
    ],
)
def test_onnx_matches_python_model(name, model, builder, tol, pixel_data):
    P, y = pixel_data
    X = _norm(P)
    model.fit(X, y)
    proto = builder(model, MEAN, STD)
    out = _run(proto, P)
    ref = model.predict_proba(X)
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < tol
    np.testing.assert_array_equal(out.argmax(1), ref.argmax(1))
    np.testing.assert_allclose(out.sum(1), 1.0, atol=1e-5)


def test_onnx_supports_single_sample_and_batches(pixel_data):
    P, y = pixel_data
    m = MLP((8,), epochs=2).fit(_norm(P), y)
    proto = mlp_to_onnx(m, MEAN, STD)
    one, many = _run(proto, P[:1]), _run(proto, P[:7])
    np.testing.assert_allclose(one[0], many[0], atol=1e-6)


def test_adaboost_json_roundtrip_matches_python(pixel_data):
    P, y = pixel_data
    X = _norm(P)
    m = AdaBoostSAMME(n_estimators=8, max_depth=3).fit(X, y)
    spec = adaboost_to_json(m, MEAN, STD)
    out = adaboost_predict_json(spec, P)
    np.testing.assert_allclose(out, m.predict_proba(X), atol=1e-12)
    assert np.array_equal(np.array(spec["classes"])[out.argmax(1)], m.predict(X))


def _run_all(model_proto, P):
    sess = ort.InferenceSession(model_proto.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, {"pixels": P.astype(np.float32)}), strict=True))


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def test_trace_outputs_explain_the_prediction(pixel_data):
    """The extra outputs the website visualises must add up to the probabilities."""
    P, y = pixel_data
    X = _norm(P)
    n = 5

    mlp = MLP((32, 16), epochs=2).fit(X, y)
    o = _run_all(mlp_to_onnx(mlp, MEAN, STD), P[:n])
    assert o["a0"].shape == (n, 32) and o["a1"].shape == (n, 16) and (o["a0"] >= 0).all()
    np.testing.assert_allclose(_softmax(o["logits"]), o["probabilities"], atol=1e-5)

    sm = SoftmaxRegression(epochs=2).fit(X, y)
    o = _run_all(softmax_regression_to_onnx(sm, MEAN, STD), P[:n])
    assert o["weights"].shape == (784, 10) and o["x"].shape == (n, 784)
    np.testing.assert_allclose(o["x"] @ o["weights"] + o["bias"], o["logits"], atol=1e-4)
    np.testing.assert_allclose(_softmax(o["logits"]), o["probabilities"], atol=1e-5)

    nb = GaussianNB(0.5).fit(X, y)
    o = _run_all(gaussian_nb_to_onnx(nb, MEAN, STD), P[:n])
    var, mu, x = o["class_var"], o["class_mean"], o["x"].astype(np.float64)
    ll = -0.5 * (np.log(2 * np.pi * var)[None] + (x[:, None, :] - mu[None]) ** 2 / var[None])
    logits = ll.sum(2) + o["log_prior"]
    np.testing.assert_allclose(logits, o["logits"], atol=1e-2, rtol=1e-4)
    np.testing.assert_allclose(_softmax(o["logits"]), o["probabilities"], atol=1e-5)
