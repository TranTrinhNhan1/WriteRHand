import numpy as np
import pytest

from src.experiments.kfold import fold_arrays, stratified_folds


def test_folds_partition_the_pool_and_keep_class_shares():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 1003)
    folds = stratified_folds(y, k=5, seed=1)
    allidx = np.concatenate(folds)
    assert sorted(allidx.tolist()) == list(range(len(y)))  # a partition: no overlap, nothing lost
    assert max(map(len, folds)) - min(map(len, folds)) <= 10
    for f in folds:
        share = np.bincount(y[f], minlength=10) / len(f)
        np.testing.assert_allclose(share, np.bincount(y, minlength=10) / len(y), atol=0.02)
    again = stratified_folds(y, k=5, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip(folds, again, strict=True))
    assert not np.array_equal(folds[0], stratified_folds(y, k=5, seed=2)[0])


def test_k_must_be_valid():
    with pytest.raises(ValueError):
        stratified_folds(np.zeros(10, dtype=int), k=1)


def test_standardisation_uses_only_the_training_part():
    rng = np.random.default_rng(3)
    X = rng.random((200, 4)).astype(np.float32)
    y = rng.integers(0, 3, 200)
    X[:20] += (
        100  # extreme images; if they land in the held-out fold they must not shift the statistics
    )
    folds = stratified_folds(y, k=4, seed=0)
    for i in range(4):
        Xtr, ytr, Xva, yva, Xte, mean, std = fold_arrays(X, y, X[:5], folds, i)
        tr = np.setdiff1d(np.arange(200), folds[i])
        assert len(ytr) == len(tr) and len(yva) == len(folds[i])
        assert mean == pytest.approx(float(X[tr].mean(dtype=np.float64)))
        assert std == pytest.approx(float(X[tr].std(dtype=np.float64, ddof=1)))
        assert abs(Xtr.mean()) < 1e-3
