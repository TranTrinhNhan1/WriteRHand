import numpy as np
import pytest
import torch

from src.models.base import BaseClassifier
from src.models.mlp import MLP


def _tiny(dtype="float64", **kw):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 6))
    y = rng.integers(0, 3, 10)
    m = MLP((7, 5), seed=3, dtype=dtype, early_stopping=False, **kw)
    m._init_params(6, 3)
    # non-zero biases make the bias gradients non-trivial
    m.b = [b + 0.1 * torch.randn(b.shape, generator=m.generator, dtype=b.dtype) for b in m.b]
    return m, X, y


def _numeric_grads(m, X, y, eps=1e-6, reseed=None):
    """Central finite differences of the loss w.r.t. every parameter."""
    out = []
    for params in (m.W, m.b):
        out.append([])
        for p in params:
            g = torch.zeros_like(p)
            flat, gflat = p.view(-1), g.view(-1)
            for j in range(flat.numel()):
                old = flat[j].item()
                vals = []
                for sgn in (+1, -1):
                    flat[j] = old + sgn * eps
                    if reseed is not None:
                        m.generator.manual_seed(reseed)
                    vals.append(m.loss_and_grads(X, y, training=reseed is not None)[0])
                flat[j] = old
                gflat[j] = (vals[0] - vals[1]) / (2 * eps)
            out[-1].append(g)
    return out


def test_gradients_match_finite_differences():
    m, X, y = _tiny()
    _, grads = m.loss_and_grads(X, y)
    num_W, num_b = _numeric_grads(m, X, y)
    for (gW, gb), nW, nb in zip(grads, num_W, num_b, strict=True):
        torch.testing.assert_close(gW, nW, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(gb, nb, atol=1e-7, rtol=1e-5)


def test_gradients_with_dropout_match_finite_differences():
    """Same dropout mask on every evaluation (generator re-seeded) -> gradient is exact."""
    m, X, y = _tiny(p_drop=0.4)
    m.generator.manual_seed(11)
    _, grads = m.loss_and_grads(X, y, training=True)
    num_W, num_b = _numeric_grads(m, X, y, reseed=11)
    for (gW, gb), nW, nb in zip(grads, num_W, num_b, strict=True):
        torch.testing.assert_close(gW, nW, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(gb, nb, atol=1e-7, rtol=1e-5)


def test_manual_backprop_equals_autograd():
    m, X, y = _tiny(p_drop=0.3)
    m.generator.manual_seed(5)
    _, grads = m.loss_and_grads(X, y, training=True)

    params = [p.clone().requires_grad_(True) for pair in zip(m.W, m.b, strict=True) for p in pair]
    m.generator.manual_seed(5)
    Xt = torch.as_tensor(X, dtype=torch.float64)
    a, last = Xt, len(m.W) - 1
    for i in range(len(m.W)):
        z = a @ params[2 * i] + params[2 * i + 1]
        if i < last:
            a = torch.relu(z)
            keep = (torch.rand(a.shape, generator=m.generator, dtype=a.dtype) >= m.p_drop).to(
                a.dtype
            )
            a = a * keep / (1 - m.p_drop)
        else:
            a = z
    torch.nn.functional.cross_entropy(a, torch.as_tensor(y)).backward()
    for i, (gW, gb) in enumerate(grads):
        torch.testing.assert_close(gW, params[2 * i].grad)
        torch.testing.assert_close(gb, params[2 * i + 1].grad)


def test_relu_derivative_is_zero_for_inactive_units():
    m = MLP((3,), dtype="float64", early_stopping=False)
    m._init_params(2, 2)
    m.W[0] = torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    m.b[0] = torch.zeros(3, dtype=torch.float64)
    X = np.array([[1.0, 0.0]])  # pre-activations: [1, -1, 0] -> unit 1 and 2 inactive
    _, grads = m.loss_and_grads(X, np.array([0]))
    assert grads[0][0][:, 1].abs().sum() == 0 and grads[0][0][:, 2].abs().sum() == 0


def test_sgd_momentum_update_matches_torch_optim():
    m, X, y = _tiny()
    ref = [p.clone().requires_grad_(True) for pair in zip(m.W, m.b, strict=True) for p in pair]
    opt = torch.optim.SGD(ref, lr=0.05, momentum=0.9)
    m.learning_rate, m.momentum = 0.05, 0.9
    vel = [(torch.zeros_like(W), torch.zeros_like(b)) for W, b in zip(m.W, m.b, strict=True)]
    for _ in range(4):
        _, grads = m.loss_and_grads(X, y)
        m._sgd_step(grads, vel)
        for i, (gW, gb) in enumerate(grads):
            ref[2 * i].grad, ref[2 * i + 1].grad = gW.clone(), gb.clone()
        opt.step()
    for i in range(len(m.W)):
        torch.testing.assert_close(m.W[i], ref[2 * i].detach())
        torch.testing.assert_close(m.b[i], ref[2 * i + 1].detach())


def test_he_initialisation_scale():
    m = MLP((512,), seed=0)
    m._init_params(784, 10)
    assert m.W[0].std().item() == pytest.approx((2 / 784) ** 0.5, rel=0.03)
    assert m.W[1].std().item() == pytest.approx((2 / 512) ** 0.5, rel=0.1)
    assert all(b.abs().sum() == 0 for b in m.b)


def test_dropout_mask_is_inverted_and_only_in_training():
    m = MLP((2000,), p_drop=0.5, dtype="float64")
    m._init_params(4, 3)
    X = torch.ones(1, 4, dtype=torch.float64)
    _, masks, _ = m._forward_cache(X, training=True)
    mask = masks[0]
    assert set(mask.unique().tolist()) == {0.0, 2.0}  # 1/(1-p) = 2
    assert mask.mean().item() == pytest.approx(1.0, abs=0.1)  # expectation preserved
    assert masks[1] is None  # output layer has no dropout
    _, masks_eval, _ = m._forward_cache(X, training=False)
    assert masks_eval[0] is None


def test_training_learns_digits_and_is_reproducible(digits):
    Xtr, ytr, Xte, yte = digits
    kw = dict(
        hidden_sizes=(32,), learning_rate=0.05, momentum=0.9, epochs=12, batch_size=32, seed=7
    )
    a = MLP(**kw).fit(Xtr, ytr, Xte, yte)
    b = MLP(**kw).fit(Xtr, ytr)
    assert a.score(Xte, yte) > 0.93
    for Wa, Wb in zip(a.W, b.W, strict=True):
        torch.testing.assert_close(Wa, Wb)  # same seed -> bit-identical weights
    assert len(a.history["val_acc"]) == a.stopped_epoch_
    c = MLP(**{**kw, "seed": 8}).fit(Xtr, ytr)
    assert not torch.equal(a.W[0], c.W[0])


def test_early_stopping_uses_internal_holdout_and_restores_best(digits):
    Xtr, ytr, Xte, yte = digits
    m = MLP((64,), learning_rate=0.2, epochs=200, patience=2, val_fraction=0.1, seed=0)
    m.fit(Xtr, ytr, Xte, yte)
    assert m.stopped_epoch_ < 200
    assert m.best_epoch_ <= m.stopped_epoch_
    assert len(m.stop_val_idx) == int(len(Xtr) * 0.1)
    assert m.best_inner_loss_ == pytest.approx(min(m.history["inner_val_loss"]))
    # the explicit validation set never influenced the stopping rule:
    m2 = MLP((64,), learning_rate=0.2, epochs=200, patience=2, val_fraction=0.1, seed=0).fit(
        Xtr, ytr
    )
    assert m2.stopped_epoch_ == m.stopped_epoch_ and m2.best_epoch_ == m.best_epoch_


def test_probabilities_and_inference_correctness(digits):
    Xtr, ytr, Xte, _ = digits
    m = MLP((16,), epochs=3).fit(Xtr, ytr)
    P = m.predict_proba(Xte)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(m.classes[P.argmax(1)], m.predict(Xte))
    np.testing.assert_allclose(np.exp(m.predict_log_proba(Xte)), P, atol=1e-6)
    # independent re-implementation of the forward pass in NumPy
    a = Xte.astype(np.float32)
    for i, (W, b) in enumerate(m.weights_numpy()):
        a = a @ W + b
        if i < len(m.W) - 1:
            a = np.maximum(a, 0)
    np.testing.assert_array_equal(a.argmax(1), m.predict(Xte))


def test_save_load_roundtrip(digits, tmp_path):
    Xtr, ytr, Xte, _ = digits
    m = MLP((16, 8), epochs=3, momentum=0.5).fit(Xtr, ytr)
    m.save(tmp_path / "mlp.npz")
    m2 = BaseClassifier.load(tmp_path / "mlp.npz")
    assert isinstance(m2, MLP) and m2.hidden_sizes == (16, 8) and m2.momentum == 0.5
    np.testing.assert_array_equal(m.predict_proba(Xte), m2.predict_proba(Xte))


def test_validation_errors(digits):
    Xtr, ytr, Xte, _ = digits
    for bad in (
        dict(p_drop=1.0),
        dict(momentum=1.0),
        dict(learning_rate=0),
        dict(hidden_sizes=(0,)),
    ):
        with pytest.raises(ValueError):
            MLP(**bad)
    with pytest.raises(RuntimeError):
        MLP().predict(Xte)
    with pytest.raises(ValueError):
        MLP().fit(Xtr, ytr[:-1])
    m = MLP((4,), epochs=1).fit(Xtr, ytr)
    with pytest.raises(ValueError):
        m.predict(Xte[:, :5])
    bad = Xtr.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        MLP((4,), epochs=1).fit(bad, ytr)
