"""Shared fixtures. Tests run offline: the 8x8 ``digits`` data bundled with scikit-learn stands in
for MNIST (same task, 64 instead of 784 features), so CI never downloads or trains on MNIST."""

import numpy as np
import pytest
from sklearn.datasets import load_digits


@pytest.fixture(scope="session")
def digits():
    d = load_digits()
    X = d.data / 16.0
    X = (X - X.mean()) / X.std()
    y = d.target
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    n_tr = 1200
    tr, te = perm[:n_tr], perm[n_tr:]
    return X[tr], y[tr], X[te], y[te]


@pytest.fixture
def blobs():
    """Three well separated Gaussian blobs in 5-D."""
    rng = np.random.default_rng(1)
    centers = np.array([[0, 0, 0, 0, 0], [4, 4, 0, 0, 0], [0, 4, 4, 0, 0]], dtype=float)
    X = np.vstack([c + rng.normal(size=(60, 5)) for c in centers])
    y = np.repeat([0, 1, 2], 60)
    return X, y
