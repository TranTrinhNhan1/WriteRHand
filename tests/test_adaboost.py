import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src.models.adaboost import AdaBoostSAMME, WeightedTree
from src.models.base import BaseClassifier, NotFittedError


def _fit_tree(X, y, w, depth, n_bins=16):
    lo, hi = X.min(0), X.max(0)
    step = (hi - lo) / n_bins
    step[step == 0] = 1
    Xb = np.empty(X.shape, dtype=np.uint8, order="F")
    Xb[:] = np.clip(np.floor((X - lo) / step), 0, n_bins - 1)
    classes, yi = np.unique(y, return_inverse=True)
    t = WeightedTree(depth, n_bins).fit_binned(
        Xb, yi, w, len(classes), np.arange(X.shape[1]), lo, step
    )
    return t, classes


def test_stump_finds_the_same_split_as_sklearn_on_weighted_data():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 16, size=(300, 6)).astype(float)  # integer grid -> bins are exact
    y = ((X[:, 2] >= 9) ^ (rng.random(300) < 0.1)).astype(int)
    w = rng.random(300)
    w /= w.sum()
    t, _ = _fit_tree(X, y, w, depth=1, n_bins=16)
    ref = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=w)
    assert t.feature[0] == ref.tree_.feature[0] == 2
    # same partition of the training set (threshold values may differ inside the gap)
    np.testing.assert_array_equal(t.predict_idx(X), ref.predict(X))


@pytest.mark.parametrize("depth", [2, 3])
def test_tree_predictions_match_sklearn_gini_tree(depth):
    rng = np.random.default_rng(depth)
    X = rng.integers(0, 16, size=(400, 5)).astype(float)
    y = (X[:, 0] >= 8).astype(int) + 2 * (X[:, 3] >= 5).astype(int)
    w = np.full(400, 1 / 400)
    t, classes = _fit_tree(X, y, w, depth)
    ref = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y, sample_weight=w)
    np.testing.assert_array_equal(classes[t.predict_idx(X)], ref.predict(X))
    assert (classes[t.predict_idx(X)] == y).all()


def test_tree_respects_sample_weights():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    heavy = np.array([0.01, 0.01, 0.97, 0.01])  # the sample (2, class 0) dominates
    t, classes = _fit_tree(X, y, heavy, depth=1, n_bins=4)
    assert classes[t.predict_idx(np.array([[2.0]]))][0] == 0
    assert t.value[0].argmax() == 0  # root majority is decided by weight, not by count


def test_samme_weight_update_formulas(digits):
    Xtr, ytr, *_ = digits
    m = AdaBoostSAMME(n_estimators=5, max_depth=2, seed=0).fit(Xtr, ytr)
    K = len(m.classes)
    err = np.array(m.history["train_err_weighted"])
    np.testing.assert_allclose(m.alphas, np.log((1 - err) / err) + np.log(K - 1))
    assert (err < 1 - 1 / K).all() and (m.alphas > 0).all()


def test_boosting_improves_over_one_tree_and_over_chance(digits):
    Xtr, ytr, Xte, yte = digits
    m = AdaBoostSAMME(n_estimators=40, max_depth=3, seed=0).fit(Xtr, ytr, Xte, yte)
    curve = m.staged_accuracy(Xte, yte, [1, 10, 40])
    assert curve[40] > curve[1] + 0.2 and curve[40] > 0.8
    np.testing.assert_allclose(curve[40], m.score(Xte, yte))
    assert m.history["val_acc"][-1] == pytest.approx(curve[40])


def test_comparable_to_sklearn_adaboost(digits):
    """Same ensemble size and weak-learner depth -> accuracy within a few points of sklearn."""
    Xtr, ytr, Xte, yte = digits
    ours = AdaBoostSAMME(n_estimators=50, max_depth=3, n_bins=32, seed=0).fit(Xtr, ytr)
    ref = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=50, random_state=0)
    ref.fit(Xtr, ytr)
    a, b = ours.score(Xte, yte), ref.score(Xte, yte)
    assert abs(a - b) < 0.06, (a, b)


def test_probabilities_are_vote_shares(digits):
    Xtr, ytr, Xte, _ = digits
    m = AdaBoostSAMME(n_estimators=10, max_depth=2).fit(Xtr, ytr)
    P = m.predict_proba(Xte)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    np.testing.assert_array_equal(m.classes[P.argmax(1)], m.predict(Xte))


def test_reproducible_and_feature_subsampling(digits):
    Xtr, ytr, Xte, _ = digits
    kw = dict(n_estimators=8, max_depth=2, max_features=0.5, seed=3)
    a, b = AdaBoostSAMME(**kw).fit(Xtr, ytr), AdaBoostSAMME(**kw).fit(Xtr, ytr)
    np.testing.assert_array_equal(a.tree_feature, b.tree_feature)
    np.testing.assert_array_equal(a.alphas, b.alphas)
    c = AdaBoostSAMME(**{**kw, "seed": 4}).fit(Xtr, ytr)
    assert not np.array_equal(a.tree_feature, c.tree_feature)


def test_save_load_roundtrip(digits, tmp_path):
    Xtr, ytr, Xte, _ = digits
    m = AdaBoostSAMME(n_estimators=6, max_depth=2).fit(Xtr, ytr)
    m.save(tmp_path / "ada.npz")
    m2 = BaseClassifier.load(tmp_path / "ada.npz")
    assert isinstance(m2, AdaBoostSAMME)
    np.testing.assert_array_equal(m.decision_scores(Xte), m2.decision_scores(Xte))


def test_validation_errors(digits):
    Xtr, ytr, Xte, _ = digits
    with pytest.raises(ValueError):
        AdaBoostSAMME(n_estimators=0)
    with pytest.raises(ValueError):
        AdaBoostSAMME(max_features=0)
    with pytest.raises(ValueError):
        AdaBoostSAMME().fit(Xtr, np.zeros(len(Xtr)))  # a single class
    with pytest.raises(NotFittedError):
        AdaBoostSAMME().predict(Xte)
    m = AdaBoostSAMME(n_estimators=2).fit(Xtr, ytr)
    assert m.n_estimators_ == 2
