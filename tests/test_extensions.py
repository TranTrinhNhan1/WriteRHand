"""Advanced extensions: extra datasets, CNN search, training replay, comparison helpers."""

import json

import numpy as np
import pytest
import torch

from src.experiments import compare, replay, search
from src.models.mlp import MLP
from src.preprocessing import preprocess as pp


def fake_raw(n_train, n_test, seed=0):
    g = np.random.default_rng(seed)

    def make(n):
        y = np.arange(n) % 10
        X = (g.random((n, 28, 28)) * 40).astype(np.uint8)
        X[np.arange(n), y, :] = 255  # class k lights up row k
        return torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))

    (Xtr, ytr), (Xte, yte) = make(n_train), make(n_test)
    return Xtr, ytr, Xte, yte


def test_dataset_registry_is_consistent():
    assert set(pp.DATASETS) == {"mnist", "fashion_mnist", "emnist_digits"}
    for spec in pp.DATASETS.values():
        assert len(spec["classes"]) == pp.N_CLASSES  # same output size for every model
    assert pp.DATASETS["mnist"]["classes"] == pp.DATASETS["emnist_digits"]["classes"]
    assert pp.DATASETS["fashion_mnist"]["classes"] != pp.DATASETS["mnist"]["classes"]
    with pytest.raises(ValueError):
        pp.check_dataset("cifar10")


def test_larger_pool_is_subsampled_and_splits_are_disjoint_and_stratified(monkeypatch):
    raw = fake_raw(1000, 300)
    monkeypatch.setattr(pp, "load_raw", lambda name, root=None: raw)
    data, meta = pp.build_dataset(name="emnist_digits", n_train=400, n_val=100)
    assert len(data["y_train"]) == 400 and len(data["y_val"]) == 100 and len(data["y_test"]) == 300
    assert meta["dataset"] == "emnist_digits" and "pool_fingerprint" in meta
    assert (np.bincount(data["y_val"], minlength=10) == 10).all()  # stratified
    # normalisation statistics come from the training portion only
    px = data["X_train"] * (meta["std"] + 1e-8) + meta["mean"]
    assert px.mean() == pytest.approx(meta["mean"], abs=1e-3)
    # same seed -> same split; different dataset name only changes metadata
    again, meta2 = pp.build_dataset(name="emnist_digits", n_train=400, n_val=100)
    assert meta2["split_fingerprint"] == meta["split_fingerprint"]
    np.testing.assert_array_equal(again["y_train"], data["y_train"])


def test_cache_paths_keep_mnist_layout(tmp_path):
    assert pp._cache_paths(tmp_path, "mnist")[0].name == "mnist.npz"
    assert pp._cache_paths(tmp_path, "fashion_mnist")[0] == tmp_path / "fashion_mnist" / "data.npz"


def test_cnn_search_runs_on_784_features_without_test_data(tmp_path, monkeypatch):
    g = np.random.default_rng(0)
    X = g.normal(size=(120, 784)).astype(np.float32)
    y = np.arange(120) % 4
    X[np.arange(120), y * 50] += 4  # separable
    data = {"X_train": X[:80], "y_train": y[:80], "X_val": X[80:], "y_val": y[80:]}
    spec = dict(search.SPECS["cnn"])
    spec["sweeps"] = [("channels", [(2,), (3,)])]
    spec["base"] = {**spec["base"], "epochs": 2, "batch_size": 40, "dense": ()}
    monkeypatch.setitem(search.SPECS, "cnn", spec)
    res = search.tune("cnn", data, out_dir=tmp_path, log=lambda *_: None)
    assert res["n_trials"] == 2 and res["best_params"]["channels"] in ([2], [3])


def test_replay_is_a_real_training_record(tmp_path):
    g = np.random.default_rng(0)
    X = g.normal(size=(300, 784)).astype(np.float32)
    y = np.arange(300) % 10
    X[np.arange(300), y * 40] += 5
    data = {"X_train": X[:200], "y_train": y[:200], "X_val": X[200:], "y_val": y[200:]}
    meta = {"mean": 0.13, "std": 0.31}
    params = {"hidden_sizes": [16], "learning_rate": 0.05, "epochs": 6, "batch_size": 50, "seed": 3}
    deployed = MLP(**{**params, "hidden_sizes": (16,)}).fit(X[:200], y[:200], X[200:], y[200:])
    deployed.save(tmp_path / "model.npz")
    rep = replay.record_replay(
        "mlp", MLP, params, data, meta, tmp_path / "model.npz", "mnist", log=lambda *_: None
    )
    assert rep["final_matches_deployed"] is True
    epochs = [s["epoch"] for s in rep["snapshots"]]
    assert epochs[0] == 0 and rep["deployed_epoch"] in epochs
    assert all(len(v) == len(rep["epochs"]) for v in rep["series"].values())
    assert all(len(s["proba"]) == len(rep["samples"]["indices"]) for s in rep["snapshots"])
    # the recorded loss series equals the model history (nothing simulated)
    np.testing.assert_allclose(rep["series"]["loss"], deployed.history["loss"], rtol=1e-5)
    # a different seed does not match the deployed weights, and the flag says so
    other = replay.record_replay(
        "mlp",
        MLP,
        {**params, "seed": 4},
        data,
        meta,
        tmp_path / "model.npz",
        "mnist",
        log=lambda *_: None,
    )
    assert other["final_matches_deployed"] is False
    json.dumps(rep)  # serialisable


def test_snapshot_epochs_and_samples():
    assert replay.snapshot_epochs(10) == [0, 1, 2, 3, 5, 8, 10]
    assert replay.snapshot_epochs(10, extra=(4,)) == [0, 1, 2, 3, 4, 5, 8, 10]
    y = np.arange(100) % 10
    idx = replay.pick_samples(y)
    assert len(idx) == replay.N_SAMPLES and set(y[idx[:10]]) == set(range(10))
    np.testing.assert_array_equal(idx, replay.pick_samples(y))


def test_mcnemar_exact_is_symmetric_and_counts_discordant_pairs():
    y = np.zeros(20, dtype=int)
    a = np.array([0] * 15 + [1] * 5)  # right on the first 15
    b = np.array([1] * 5 + [0] * 15)  # right on the last 15
    only_a, only_b, p = compare.mcnemar_exact(a, b, y)
    assert (only_a, only_b) == (5, 5) and p == pytest.approx(1.0)
    only_a, only_b, p = compare.mcnemar_exact(np.zeros(20, int), np.ones(20, int), y)
    assert (only_a, only_b) == (20, 0) and p < 1e-5


def test_cross_eval_refuses_different_class_definitions():
    with pytest.raises(ValueError):
        compare.cross_eval("mnist", "fashion_mnist", ".")
