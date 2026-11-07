import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from src.models.base import BaseClassifier, NotFittedError
from src.models.logistic_regression import SoftmaxRegression, cross_entropy, log_softmax, softmax


def test_softmax_is_stable_and_normalised():
    z = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]])
    p = softmax(z)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(softmax(z), softmax(z + 12345.0))  # shift invariance
    np.testing.assert_allclose(np.exp(log_softmax(z)), p)


def test_cross_entropy_matches_definition():
    z = np.array([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    y = np.array([0, 2])
    p = softmax(z)
    assert cross_entropy(z, y) == pytest.approx(-np.mean(np.log(p[[0, 1], y])))
    assert np.isfinite(cross_entropy(np.array([[1e4, 0.0]]), np.array([1])))  # never log(0)


@pytest.mark.parametrize("l2", [0.0, 0.3])
def test_analytic_gradient_matches_finite_differences(l2):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 5))
    Y = np.eye(4)[rng.integers(0, 4, 12)]
    W, b = rng.normal(size=(5, 4)) * 0.5, rng.normal(size=4) * 0.1
    _, gW, gb = SoftmaxRegression.loss_and_grad(W, b, X, Y, l2)
    eps = 1e-6

    def f(W_, b_):
        return SoftmaxRegression.loss_and_grad(W_, b_, X, Y, l2)[0]

    num_W = np.zeros_like(W)
    for i in np.ndindex(*W.shape):
        Wp, Wm = W.copy(), W.copy()
        Wp[i] += eps
        Wm[i] -= eps
        num_W[i] = (f(Wp, b) - f(Wm, b)) / (2 * eps)
    num_b = np.array([(f(W, b + eps * e) - f(W, b - eps * e)) / (2 * eps) for e in np.eye(4)])
    np.testing.assert_allclose(gW, num_W, atol=1e-7)
    np.testing.assert_allclose(gb, num_b, atol=1e-7)


def test_loss_decreases_and_history_recorded(digits):
    Xtr, ytr, Xte, yte = digits
    m = SoftmaxRegression(learning_rate=0.1, epochs=15, batch_size=64).fit(Xtr, ytr, Xte, yte)
    loss = m.history["loss"]
    assert loss[-1] < loss[0] * 0.3
    assert len(m.history["val_acc"]) == len(loss) == m.n_iter_
    assert m.score(Xte, yte) > 0.93


def test_early_stopping_on_tolerance(digits):
    Xtr, ytr, *_ = digits
    m = SoftmaxRegression(0.1, epochs=500, tol=1e-3, n_iter_no_change=2).fit(Xtr, ytr)
    assert m.n_iter_ < 500


def test_matches_sklearn_when_both_converge(digits):
    """Full-batch GD on the L2-regularised loss reaches the unique optimum found by L-BFGS."""
    Xtr, ytr, Xte, _ = digits
    l2 = 0.05
    ours = SoftmaxRegression(learning_rate=0.5, l2=l2, epochs=6000, batch_size=None).fit(Xtr, ytr)
    # sklearn minimises C * sum(CE) + 0.5||W||^2  <=>  mean CE + (1 / (2 C n)) ||W||^2
    ref = SkLogisticRegression(C=1.0 / (l2 * len(Xtr)), tol=1e-10, max_iter=5000).fit(Xtr, ytr)
    Y = np.eye(10)[ytr]
    obj_ours = SoftmaxRegression.loss_and_grad(ours.W, ours.b, Xtr, Y, l2)[0]
    obj_ref = SoftmaxRegression.loss_and_grad(ref.coef_.T, ref.intercept_, Xtr, Y, l2)[0]
    assert obj_ours == pytest.approx(obj_ref, abs=1e-5)  # same optimum of the same objective
    np.testing.assert_allclose(ours.W, ref.coef_.T, atol=2e-3)
    np.testing.assert_allclose(ours.b, ref.intercept_, atol=5e-3)
    assert np.mean(ours.predict(Xte) == ref.predict(Xte)) > 0.995
    np.testing.assert_allclose(ours.predict_proba(Xte), ref.predict_proba(Xte), atol=5e-3)


def test_reproducible_and_seed_sensitive(digits):
    Xtr, ytr, *_ = digits
    a = SoftmaxRegression(epochs=3, seed=1).fit(Xtr, ytr)
    b = SoftmaxRegression(epochs=3, seed=1).fit(Xtr, ytr)
    c = SoftmaxRegression(epochs=3, seed=2).fit(Xtr, ytr)
    np.testing.assert_array_equal(a.W, b.W)
    assert not np.array_equal(a.W, c.W)


def test_probabilities_sum_to_one_and_agree_with_predict(digits):
    Xtr, ytr, Xte, _ = digits
    m = SoftmaxRegression(epochs=5).fit(Xtr, ytr)
    P = m.predict_proba(Xte)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    np.testing.assert_array_equal(m.classes[P.argmax(1)], m.predict(Xte))


def test_huge_learning_rate_does_not_produce_nan(digits):
    """Stable softmax / log-sum-exp: a silly step size gives a bad model, never NaN probabilities."""
    Xtr, ytr, Xte, _ = digits
    m = SoftmaxRegression(learning_rate=1e6, epochs=3).fit(Xtr, ytr)
    assert np.isfinite(m.predict_proba(Xte)).all()


def test_validation_and_roundtrip(digits, tmp_path):
    Xtr, ytr, Xte, _ = digits
    with pytest.raises(ValueError):
        SoftmaxRegression(learning_rate=0)
    with pytest.raises(NotFittedError):
        SoftmaxRegression().predict(Xte)
    m = SoftmaxRegression(epochs=3, l2=1e-3).fit(Xtr, ytr)
    with pytest.raises(ValueError):
        m.predict(Xte[:, :10])
    m.save(tmp_path / "lr.npz")
    m2 = BaseClassifier.load(tmp_path / "lr.npz")
    np.testing.assert_array_equal(m.predict_proba(Xte), m2.predict_proba(Xte))
    assert m2.get_params() == m.get_params()
