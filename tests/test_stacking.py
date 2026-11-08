import numpy as np
import pytest

from src.models.logistic_regression import SoftmaxRegression
from src.models.naive_bayes import GaussianNB
from src.models.stacking import StackingEnsemble, base_scores
from src.models.svm import LinearSVM


@pytest.fixture
def fitted(digits):
    Xtr, ytr, Xte, yte = digits
    n = len(Xte) // 2
    bases = [
        ("GaussianNB", GaussianNB(var_smoothing=0.1).fit(Xtr, ytr)),
        ("SoftmaxRegression", SoftmaxRegression(0.1, epochs=10).fit(Xtr, ytr)),
        ("LinearSVM", LinearSVM(0.01, epochs=10).fit(Xtr, ytr)),
    ]
    return bases, (Xte[:n], yte[:n]), (Xte[n:], yte[n:])


def test_features_use_log_proba_or_svm_scores(fitted):
    bases, (Xv, _), _ = fitted
    nb, sm, svm = (m for _, m in bases)
    np.testing.assert_allclose(base_scores(sm, Xv), np.log(np.clip(sm.predict_proba(Xv), 1e-4, 1)))
    np.testing.assert_allclose(base_scores(svm, Xv), svm.decision_function(Xv))
    st = StackingEnsemble(bases=bases)
    assert st.features(Xv).shape == (len(Xv), 30)


def test_fit_on_holdout_predicts_better_than_chance_and_is_consistent(fitted):
    bases, (Xv, yv), (Xt, yt) = fitted
    st = StackingEnsemble(bases=bases, l2=1e-3, epochs=30).fit(Xv, yv)
    assert st.score(Xt, yt) > 0.9
    np.testing.assert_array_equal(st.predict(Xt), st.predict_from_features(st.features(Xt)))
    p = st.predict_proba(Xt)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    imp = st.block_importance()
    assert set(imp) == {"GaussianNB", "SoftmaxRegression", "LinearSVM"} and all(
        v > 0 for v in imp.values()
    )


def test_save_and_reload_with_the_same_bases(fitted, tmp_path):
    bases, (Xv, yv), (Xt, _) = fitted
    st = StackingEnsemble(bases=bases, epochs=10).fit(Xv, yv)
    path = st.save(tmp_path / "stack.npz")
    st2 = StackingEnsemble.load_with_bases(path, bases)
    np.testing.assert_array_equal(st.predict(Xt), st2.predict(Xt))
    with pytest.raises(ValueError):
        StackingEnsemble.load_with_bases(path, bases[::-1])


def test_cross_validated_estimate_is_computed_only_inside_the_holdout(fitted):
    from src.experiments.stacking import cv_accuracy

    bases, (Xv, yv), _ = fitted
    F = StackingEnsemble(bases=bases).features(Xv)
    mean, sd = cv_accuracy(F, yv, 1e-3, folds=3)
    assert 0.5 < mean <= 1.0 and sd >= 0.0
