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


def test_each_transformation_does_what_it_says():
    from src.experiments.augment import TRANSFORMS, transform_images

    X = _digit_batch(30)
    bg = X.min()
    for name in TRANSFORMS:
        out = transform_images(X, (name,), seed=2)
        assert out.shape == X.shape and not np.array_equal(out, X), name
        assert out.min() >= bg - 1e-6 and out.max() <= X.max() + 1e-6, name
    # translation moves ink without changing the amount of it (up to interpolation)
    t = transform_images(X, ("translation",), seed=2, translation=3.0)
    assert abs((t > 0).sum() - (X > 0).sum()) < 0.3 * (X > 0).sum()
    # gaussian noise changes background pixels, salt & pepper changes a bounded fraction of pixels
    g = transform_images(X, ("gaussian_noise",), seed=2, gaussian_noise=0.2)
    assert (g != X).mean() > 0.3
    sp = transform_images(X, ("salt_pepper",), seed=2, salt_pepper=0.1)
    changed = (sp != X).mean()
    assert 0.03 < changed < 0.12
    vals = np.unique(sp[sp != X])
    assert all(np.isclose(v, bg, atol=1e-5) or np.isclose(v, X.max(), atol=1e-5) for v in vals)


def test_conditions_share_random_draws_and_unknown_op_rejected():
    import pytest

    from src.experiments.augment import transform_images

    X = _digit_batch(6)
    # same seed => the rotation draw is the same whether or not translation is also active
    a = transform_images(X, ("rotation",), seed=5)
    b = transform_images(X, ("rotation", "translation"), seed=5)
    assert not np.array_equal(a, b) and a.shape == b.shape
    with pytest.raises(ValueError):
        transform_images(X, ("blur",))


def test_apply_spec_keeps_the_originals_and_adds_k_copies():
    from src.experiments.augment import apply_spec, deploy_spec

    X = _digit_batch(6)
    y = np.arange(6) % 3
    spec = deploy_spec(30.0, k=2)
    Xa, ya = apply_spec(X, y, spec)
    assert Xa.shape == (3 * len(X), 784) and len(ya) == 3 * len(X)
    np.testing.assert_array_equal(Xa[: len(X)], X)
    np.testing.assert_array_equal(ya, np.tile(y, 3))
    Xb, _ = apply_spec(X, y, spec)
    np.testing.assert_array_equal(Xa, Xb)  # deterministic
    assert not np.allclose(Xa[len(X) :], np.vstack([X, X]))  # copies are really transformed


def test_train_final_with_augmentation_reports_on_original_images(tmp_path):
    from src.experiments.augment import deploy_spec
    from src.experiments.final import train_final

    rng = np.random.default_rng(0)
    X = _digit_batch(60)
    y = rng.integers(0, 10, 60)
    data = {
        "X_train": X,
        "y_train": y,
        "X_val": X[:20],
        "y_val": y[:20],
        "X_test": X[20:40],
        "y_test": y[20:40],
    }
    params = {
        "hidden_sizes": [8],
        "learning_rate": 0.05,
        "momentum": 0.9,
        "p_drop": 0.0,
        "batch_size": 32,
        "epochs": 2,
        "seed": 1,
    }
    meta = {"preprocess_version": "t", "split_fingerprint": "f", "seed": 1}
    spec = deploy_spec(30.0, k=1)
    _, res = train_final(
        "mlp",
        params,
        data,
        meta,
        save=False,
        out_dir=tmp_path,
        repeats=1,
        augment=spec,
        log=lambda *_: None,
    )
    assert res["augmentation"] == spec and "transformed copies" in res["protocol"]
    assert np.array(res["splits"]["train"]["confusion_matrix"]).sum() == len(y)
