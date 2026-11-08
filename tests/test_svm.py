import numpy as np
import pytest
from sklearn.svm import LinearSVC

from src.models.base import BaseClassifier, NotFittedError
from src.models.svm import LinearSVM, hinge_terms


def test_hinge_terms_match_definition():
    Z = np.array([[2.0, -0.5, 0.3]])
    T = np.array([[1.0, -1.0, 1.0]])
    h, dh = hinge_terms(Z, T, "hinge")
    np.testing.assert_allclose(h, [[0.0, 0.5, 0.7]])  # 1 - t z, clipped at 0
    # the derivative is -t where the margin is violated
    np.testing.assert_allclose(dh, [[0.0, 1.0, -1.0]])
    s, ds = hinge_terms(Z, T, "squared_hinge")
    np.testing.assert_allclose(s, [[0.0, 0.25, 0.49]])
    np.testing.assert_allclose(ds, [[0.0, 1.0, -1.4]])


@pytest.mark.parametrize("l2", [0.0, 0.3])
def test_squared_hinge_gradient_matches_finite_differences(l2):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 5))
    Y = np.eye(4)[rng.integers(0, 4, 12)]
    W, b = rng.normal(size=(5, 4)) * 0.5, rng.normal(size=4) * 0.1
    _, gW, gb = LinearSVM.loss_and_grad(W, b, X, Y, l2, "squared_hinge")
    eps = 1e-6

    def f(W_, b_):
        return LinearSVM.loss_and_grad(W_, b_, X, Y, l2, "squared_hinge")[0]

    num_W = np.zeros_like(W)
    for i in np.ndindex(*W.shape):
        Wp, Wm = W.copy(), W.copy()
        Wp[i] += eps
        Wm[i] -= eps
        num_W[i] = (f(Wp, b) - f(Wm, b)) / (2 * eps)
    num_b = np.array([(f(W, b + eps * e) - f(W, b - eps * e)) / (2 * eps) for e in np.eye(4)])
    np.testing.assert_allclose(gW, num_W, atol=1e-6)
    np.testing.assert_allclose(gb, num_b, atol=1e-6)


def test_hinge_subgradient_is_a_descent_direction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 6))
    Y = np.eye(3)[rng.integers(0, 3, 30)]
    W, b = rng.normal(size=(6, 3)), np.zeros(3)
    f0, gW, gb = LinearSVM.loss_and_grad(W, b, X, Y, 0.01, "hinge")
    f1 = LinearSVM.loss_and_grad(W - 1e-3 * gW, b - 1e-3 * gb, X, Y, 0.01, "hinge")[0]
    assert f1 < f0


@pytest.mark.parametrize("loss", ["hinge", "squared_hinge"])
def test_learns_digits_and_records_history(loss, digits):
    Xtr, ytr, Xte, yte = digits
    m = LinearSVM(learning_rate=0.01, l2=1e-4, loss=loss, epochs=20, batch_size=64).fit(
        Xtr, ytr, Xte, yte
    )
    assert m.history["loss"][-1] < m.history["loss"][0]
    assert len(m.history["val_acc"]) == len(m.history["loss"]) == m.n_iter_
    assert m.score(Xte, yte) > 0.9
    assert m.decision_function(Xte).shape == (len(yte), 10)


def test_close_to_liblinear_on_digits(digits):
    """Same objective (C = 1 / (l2 n)); liblinear also penalises the intercept, so only roughly equal."""
    Xtr, ytr, Xte, yte = digits
    ours = LinearSVM(
        learning_rate=0.02, l2=1e-3, loss="squared_hinge", epochs=60, batch_size=64
    ).fit(Xtr, ytr)
    sk = LinearSVC(C=1.0 / (1e-3 * len(ytr)), loss="squared_hinge", dual=False, max_iter=5000).fit(
        Xtr, ytr
    )
    assert abs(ours.score(Xte, yte) - sk.score(Xte, yte)) < 0.03
    assert np.mean(ours.predict(Xte) == sk.predict(Xte)) > 0.93


def test_save_load_roundtrip_and_errors(digits, tmp_path):
    Xtr, ytr, Xte, _ = digits
    m = LinearSVM(epochs=3).fit(Xtr, ytr)
    path = m.save(tmp_path / "svm.npz")
    m2 = BaseClassifier.load(path)
    assert isinstance(m2, LinearSVM) and m2.loss == m.loss
    np.testing.assert_array_equal(m.predict(Xte), m2.predict(Xte))
    with pytest.raises(NotFittedError):
        LinearSVM().predict(Xte)
    with pytest.raises(ValueError):
        LinearSVM(loss="log")
    with pytest.raises(ValueError):
        LinearSVM(learning_rate=0)
