"""CNN: hand-written convolution / pooling / gradients checked against independent references."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models import cnn as C
from src.models.base import BaseClassifier
from src.models.cnn import CNN


def naive_conv(x, K, b):
    """Textbook 7-loop cross-correlation in float64 NumPy (independent of the im2col code)."""
    B, Cin, H, W = x.shape
    Cout, _, k, _ = K.shape
    Ho, Wo = H - k + 1, W - k + 1
    y = np.zeros((B, Cout, Ho, Wo))
    for n in range(B):
        for o in range(Cout):
            for r in range(Ho):
                for c in range(Wo):
                    s = b[o]
                    for ci in range(Cin):
                        for i in range(k):
                            for j in range(k):
                                s += x[n, ci, r + i, c + j] * K[o, ci, i, j]
                    y[n, o, r, c] = s
    return y


@pytest.fixture
def small():
    g = np.random.default_rng(0)
    x = g.normal(size=(2, 3, 9, 8))
    K = g.normal(size=(4, 3, 3, 3))
    b = g.normal(size=4)
    return x, K, b


def test_conv_matches_naive_loops_and_torch(small):
    x, K, b = small
    y, _ = C.conv_forward(torch.from_numpy(x), torch.from_numpy(K), torch.from_numpy(b))
    np.testing.assert_allclose(y.numpy(), naive_conv(x, K, b), atol=1e-12)
    ref = F.conv2d(torch.from_numpy(x), torch.from_numpy(K), torch.from_numpy(b))
    np.testing.assert_allclose(y.numpy(), ref.numpy(), atol=1e-12)


def test_maxpool_matches_torch_including_odd_border():
    x = torch.randn(2, 3, 11, 9, dtype=torch.float64)
    y, _ = C.maxpool_forward(x, 2)
    np.testing.assert_allclose(y.numpy(), F.max_pool2d(x, 2).numpy())
    assert y.shape == (2, 3, 5, 4)


def test_conv_and_pool_gradients_match_autograd(small):
    x, K, b = [torch.from_numpy(a).requires_grad_() for a in small]
    y = F.max_pool2d(torch.relu(F.conv2d(x, K, b)), 2)
    w = torch.randn_like(y)
    (y * w).sum().backward()

    with torch.no_grad():
        z, cols = C.conv_forward(x, K, b)
        a = torch.relu(z)
        p, arg = C.maxpool_forward(a, 2)
        da = C.maxpool_backward(w, arg, a.shape, 2)
        dz = da * (z > 0)
        dx, dK, db = C.conv_backward(dz, cols, K, tuple(x.shape))
    np.testing.assert_allclose(p.numpy(), y.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(dx.numpy(), x.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(dK.numpy(), K.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(db.numpy(), b.grad.numpy(), atol=1e-10)


def _model(dtype="float64", **kw):
    m = CNN(channels=(2, 3), kernel_size=3, dense=(5,), dtype=dtype, seed=1, **kw)
    m.classes = np.arange(4)
    m.n_classes = 4
    m._init_params(4)
    return m


def test_full_network_gradients_match_autograd_and_finite_differences():
    m = _model()
    g = np.random.default_rng(1)
    X = g.normal(size=(6, 784))
    y = g.integers(0, 4, 6)
    loss, cg, dg = m.loss_and_grads(X, y)

    params = [p.clone().requires_grad_() for p in m._params()]
    n_k, n_w = len(m.K), len(m.W)
    saved = (m.K, m.kb, m.W, m.b)
    m.K, m.kb = params[:n_k], params[n_k : 2 * n_k]
    m.W, m.b = params[2 * n_k : 2 * n_k + n_w], params[2 * n_k + n_w :]
    _, logits = m._forward_cache(torch.from_numpy(X))
    F.cross_entropy(logits, torch.from_numpy(y)).backward()
    manual = (
        [g_[0] for g_ in cg] + [g_[1] for g_ in cg] + [g_[0] for g_ in dg] + [g_[1] for g_ in dg]
    )
    for p, mg in zip(params, manual, strict=True):
        np.testing.assert_allclose(mg.numpy(), p.grad.numpy(), atol=1e-9)
    m.K, m.kb, m.W, m.b = saved

    # finite differences on a few entries of the first kernel
    eps = 1e-6
    for idx in [(0, 0, 0, 0), (1, 0, 2, 1)]:
        base = m.K[0][idx].item()
        m.K[0][idx] = base + eps
        up = m.loss_and_grads(X, y)[0]
        m.K[0][idx] = base - eps
        dn = m.loss_and_grads(X, y)[0]
        m.K[0][idx] = base
        assert (up - dn) / (2 * eps) == pytest.approx(cg[0][0][idx].item(), abs=1e-6)


def test_fit_learns_and_persistence_roundtrip(tmp_path):
    g = np.random.default_rng(0)
    # two obviously different classes: bright top half vs bright bottom half
    X = g.normal(0, 0.1, (200, 28, 28))
    y = np.repeat([0, 1], 100)
    X[y == 0, :14] += 1.0
    X[y == 1, 14:] += 1.0
    X = X.reshape(200, -1)
    m = CNN(channels=(2,), epochs=4, batch_size=32, learning_rate=0.05, seed=3).fit(X, y)
    assert m.score(X, y) > 0.95
    m.save(tmp_path / "m.npz")
    m2 = BaseClassifier.load(tmp_path / "m.npz")
    assert isinstance(m2, CNN)
    np.testing.assert_allclose(m2.predict_proba(X), m.predict_proba(X), atol=1e-6)


def test_describe_shapes_come_from_the_forward_pass():
    m = _model("float32")
    layers = {layer["id"]: layer for layer in m.describe()}
    assert layers["conv0"]["out_shape"] == [2, 26, 26]
    assert layers["pool0"]["out_shape"] == [2, 13, 13]
    assert layers["conv1"]["out_shape"] == [3, 11, 11]
    assert layers["pool1"]["out_shape"] == [3, 5, 5]
    assert layers["flat"]["out_shape"] == [75]
    assert layers["z0"]["weights"] == [75, 5] and layers["logits"]["weights"] == [5, 4]
    tr = m.forward_trace(np.zeros((3, 784)))
    assert tr["pool1"].shape == (3, 3, 5, 5) and tr["logits"].shape == (3, 4)


def test_invalid_hyper_parameters():
    with pytest.raises(ValueError):
        CNN(channels=())
    with pytest.raises(ValueError):
        CNN(channels=(4, 4, 4, 4), kernel_size=5)  # image shrinks below one pixel
    with pytest.raises(ValueError):
        CNN(learning_rate=0)
