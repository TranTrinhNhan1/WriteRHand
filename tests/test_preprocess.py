import numpy as np
import pytest
import torch

from src.models.cross_validation import create_folds, run_kfold
from src.preprocessing.preprocess import (
    flatten,
    prepare_mnist,
    split_fingerprint,
    standardize,
    stratified_split_indices,
)


def _labels(n=1000, k=10, seed=0):
    return np.random.default_rng(seed).integers(0, k, size=n)


def test_stratified_split_is_partition_and_sized():
    y = _labels()
    tr, va = stratified_split_indices(y, 200, seed=1)
    assert len(va) == 200 and len(tr) == 800
    assert len(np.intersect1d(tr, va)) == 0
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(len(y)))


def test_stratified_split_keeps_class_proportions():
    y = _labels(5000)
    _, va = stratified_split_indices(y, 1000, seed=3)
    full = np.bincount(y, minlength=10) / len(y)
    part = np.bincount(y[va], minlength=10) / len(va)
    assert np.abs(full - part).max() < 0.002


def test_split_is_deterministic_and_seed_dependent():
    y = _labels()
    a = stratified_split_indices(y, 200, seed=7)
    b = stratified_split_indices(y, 200, seed=7)
    c = stratified_split_indices(y, 200, seed=8)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])
    assert split_fingerprint(*a) == split_fingerprint(*b) != split_fingerprint(*c)


def test_split_rejects_bad_sizes():
    with pytest.raises(ValueError):
        stratified_split_indices(_labels(), 0)
    with pytest.raises(ValueError):
        stratified_split_indices(_labels(), 1000)


def _fake_mnist(n_train=600, n_test=100):
    g = torch.Generator().manual_seed(0)
    Xtr = torch.randint(0, 256, (n_train, 28, 28), generator=g, dtype=torch.uint8)
    ytr = torch.arange(n_train) % 10
    Xte = torch.randint(0, 256, (n_test, 28, 28), generator=g, dtype=torch.uint8)
    yte = torch.arange(n_test) % 10
    return Xtr, ytr, Xte, yte


def test_normalisation_statistics_come_from_train_only():
    Xtr, ytr, Xte, yte = _fake_mnist()
    Xte = (Xte // 2).to(torch.uint8)  # test distribution differs on purpose
    X_train, y_train, X_val, y_val, X_test, y_test, mean, std = prepare_mnist(
        Xtr, ytr, Xte, yte, n_train=500
    )
    raw = flatten(Xtr)
    tr_idx, _ = stratified_split_indices(ytr.numpy(), 100, seed=42)
    expected_mean = raw[torch.from_numpy(tr_idx)].mean().item()
    assert mean == pytest.approx(expected_mean, rel=1e-6)
    assert X_train.mean().abs().item() < 1e-4 and X_train.std().item() == pytest.approx(1, abs=1e-3)
    assert abs(X_test.mean().item()) > 0.05  # NOT re-centred with test statistics
    assert X_train.shape == (500, 784) and X_val.shape == (100, 784) and X_test.shape == (100, 784)
    np.testing.assert_allclose(
        X_test.numpy(), standardize(flatten(Xte), mean, std).numpy(), atol=1e-6
    )


def test_folds_partition_and_are_stratified():
    y = _labels(400)
    folds = create_folds(len(y), k=4, seed=0, y=y)
    all_val = np.concatenate([v.numpy() for _, v in folds])
    assert np.array_equal(np.sort(all_val), np.arange(400))
    for tr, va in folds:
        assert len(np.intersect1d(tr.numpy(), va.numpy())) == 0
        assert len(tr) + len(va) == 400


def test_kfold_fits_preprocessing_inside_each_fold():
    """Every fold must be normalised with its own training part (no leakage from validation)."""
    Xtr, ytr, _, _ = _fake_mnist(400)
    seen = []

    class Probe:
        def fit(self, X, y):
            seen.append((float(X.mean()), float(X.std())))
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    run_kfold(Probe, Xtr, ytr, K=4, seed=0, labels=range(10))
    assert len(seen) == 4
    for m, s in seen:
        assert abs(m) < 1e-4 and s == pytest.approx(1, abs=1e-3)
