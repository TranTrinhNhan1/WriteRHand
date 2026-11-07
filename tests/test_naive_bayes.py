import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB as SkGaussianNB

from src.models.base import BaseClassifier
from src.models.naive_bayes import GaussianNB


@pytest.mark.parametrize("smoothing", [1e-9, 1e-3, 0.1])
def test_matches_sklearn(digits, smoothing):
    Xtr, ytr, Xte, _ = digits
    ours = GaussianNB(var_smoothing=smoothing).fit(Xtr, ytr)
    ref = SkGaussianNB(var_smoothing=smoothing).fit(Xtr, ytr)
    np.testing.assert_allclose(ours.mean, ref.theta_, atol=1e-10)
    np.testing.assert_allclose(ours.var, ref.var_, rtol=1e-8)
    np.testing.assert_allclose(ours.prior, ref.class_prior_)
    np.testing.assert_array_equal(ours.predict(Xte), ref.predict(Xte))
    np.testing.assert_allclose(ours.predict_proba(Xte), ref.predict_proba(Xte), atol=1e-8)
    np.testing.assert_allclose(ours.predict_log_proba(Xte), ref.predict_log_proba(Xte), atol=1e-6)


def test_probabilities_are_normalised_and_stable(digits):
    Xtr, ytr, Xte, _ = digits
    nb = GaussianNB(1e-9).fit(Xtr, ytr)
    P = nb.predict_proba(Xte * 50)  # extreme inputs -> huge negative log-likelihoods
    assert np.isfinite(P).all()
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_constant_feature_does_not_break_fit(blobs):
    X, y = blobs
    X = np.hstack([X, np.zeros((len(X), 1))])  # zero-variance column
    nb = GaussianNB(1e-9).fit(X, y)
    assert np.isfinite(nb.var).all() and (nb.var > 0).all()
    assert nb.score(X, y) > 0.9


def test_input_validation(blobs):
    X, y = blobs
    with pytest.raises(ValueError):
        GaussianNB().fit(X, y[:-1])
    with pytest.raises(ValueError):
        GaussianNB().fit(np.full_like(X, np.nan), y)
    with pytest.raises(ValueError):
        GaussianNB().fit(X[:, 0], y)
    with pytest.raises(ValueError):
        GaussianNB(var_smoothing=-1)
    nb = GaussianNB().fit(X, y)
    with pytest.raises(ValueError):
        nb.predict(X[:, :3])


def test_save_load_roundtrip(blobs, tmp_path):
    X, y = blobs
    nb = GaussianNB(1e-3).fit(X, y)
    nb.save(tmp_path / "nb.npz")
    nb2 = BaseClassifier.load(tmp_path / "nb.npz")
    assert isinstance(nb2, GaussianNB) and nb2.var_smoothing == 1e-3
    np.testing.assert_array_equal(nb.predict_proba(X), nb2.predict_proba(X))
