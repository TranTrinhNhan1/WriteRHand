import numpy as np

from src.experiments.augment import augment_images


def _digit_batch(n=5):
    X = np.full((n, 784), -0.4, dtype=np.float32)
    img = X.reshape(n, 28, 28)
    img[:, 8:20, 12:16] = 2.0  # a vertical bar
    return X


def test_shapes_labels_and_originals_untouched():
    X = _digit_batch()
    X0 = X.copy()
    out = augment_images(X, k=3, seed=1)
    assert out.shape == (15, 784) and out.dtype == X.dtype
    np.testing.assert_array_equal(X, X0)


def test_deterministic_and_seed_dependent():
    X = _digit_batch()
    np.testing.assert_array_equal(augment_images(X, 1, seed=3), augment_images(X, 1, seed=3))
    assert not np.array_equal(augment_images(X, 1, seed=3), augment_images(X, 1, seed=4))


def test_background_preserved_and_ink_not_destroyed():
    X = _digit_batch(20)
    out = augment_images(X, 1, seed=0)
    assert out.min() >= X.min() - 1e-6 and out.max() <= X.max() + 1e-6
    ink_before = (X > 0).sum(1).mean()
    ink_after = (out > 0).sum(1).mean()
    assert 0.6 * ink_before < ink_after < 1.5 * ink_before


def test_zero_transform_is_identity():
    X = _digit_batch(3)
    out = augment_images(X, 1, seed=0, max_rot_deg=0, max_shift=0, scale=(1.0, 1.0))
    np.testing.assert_allclose(out, X, atol=1e-5)
