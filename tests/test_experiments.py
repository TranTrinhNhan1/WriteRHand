"""Smoke tests of the experiment pipeline on tiny data (seconds, CPU only)."""

import json

import numpy as np
import pytest

from src.experiments import search
from src.experiments.final import run_baseline, sklearn_counterpart, train_final
from src.experiments.tracking import Tracker, resolve_mode


@pytest.fixture
def tiny(digits):
    Xtr, ytr, Xte, yte = digits
    n = len(Xte) // 2
    return {
        "X_train": Xtr[:600],
        "y_train": ytr[:600],
        "X_val": Xte[:n],
        "y_val": yte[:n],
        "X_test": Xte[n:],
        "y_test": yte[n:],
    }


META = {"preprocess_version": "test", "split_fingerprint": "abc", "seed": 42}


@pytest.mark.parametrize("model_key", [k for k in search.SPECS if k != "cnn"])
def test_tune_needs_no_test_data_and_logs_every_trial(model_key, tiny, tmp_path, monkeypatch):
    # tiny 10-class problem with 64 features: shrink the sweeps
    sweeps = {
        "naive_bayes": [("var_smoothing", [1e-9, 0.1])],
        "softmax": [("learning_rate", [0.05, 0.2])],
        "mlp": [("hidden_sizes", [(8,), (16,)])],
        "adaboost": [("max_depth", [1, 2])],
    }
    base = {"softmax": {"epochs": 3}, "mlp": {"epochs": 3}, "adaboost": {"n_estimators": 4}}
    spec = dict(search.SPECS[model_key])
    spec["sweeps"] = sweeps[model_key]
    spec["base"] = {**spec["base"], **base.get(model_key, {})}
    if "epoch_curve" in spec:
        spec["epoch_curve"] = {**spec["epoch_curve"], "max": 6}
    monkeypatch.setitem(search.SPECS, model_key, spec)

    data = {k: v for k, v in tiny.items() if "test" not in k}  # no test arrays at all
    res = search.tune(model_key, data, out_dir=tmp_path, log=lambda *_: None)
    saved = json.loads((tmp_path / f"{search.CLASS_NAME[model_key]}.json").read_text())
    assert saved["n_trials"] == len(saved["trials"]) >= 2
    assert saved["best_params"].keys() >= set(sweeps[model_key][0][:1])
    # the chosen value is the argmax of validation accuracy among the swept values
    swept = [t for t in saved["trials"] if t["stage"].startswith("sweep")]
    best = max(swept, key=lambda t: t["val_acc"])
    assert saved["best_params"][best["param"]] == best["value"] or search.SPECS[model_key].get(
        "epoch_curve"
    )
    assert res["n_val"] == len(data["y_val"])


def test_train_final_reports_all_splits_and_saves_model(tiny, tmp_path):
    params = {"var_smoothing": 0.1}
    model, res = train_final(
        "naive_bayes",
        params,
        tiny,
        META,
        tracker=None,
        out_dir=tmp_path / "res",
        art_dir=tmp_path / "art",
        repeats=2,
        log=lambda *_: None,
    )
    assert set(res["splits"]) == {"train", "val", "test"}
    for m in res["splits"].values():
        assert {"accuracy", "precision_macro", "recall_macro", "f1_macro"} <= set(m)
        assert np.array(m["confusion_matrix"]).shape == (10, 10)
    assert (tmp_path / "res" / "GaussianNB.json").is_file()
    assert (tmp_path / "art" / "model.npz").is_file()
    assert res["train_time_sec"] > 0 and res["latency_ms_single_sample"] > 0


def test_results_are_reproducible(tiny, tmp_path):
    p = {
        "hidden_sizes": [8],
        "learning_rate": 0.05,
        "momentum": 0.9,
        "p_drop": 0.0,
        "batch_size": 32,
        "epochs": 4,
        "seed": 42,
    }
    a = train_final(
        "mlp", p, tiny, META, save=False, out_dir=tmp_path / "a", repeats=1, log=lambda *_: None
    )[1]
    b = train_final(
        "mlp", p, tiny, META, save=False, out_dir=tmp_path / "b", repeats=1, log=lambda *_: None
    )[1]
    np.testing.assert_array_equal(
        a["splits"]["test"]["confusion_matrix"], b["splits"]["test"]["confusion_matrix"]
    )
    assert a["history"]["loss"] == b["history"]["loss"]


@pytest.mark.parametrize(
    "model_key,params",
    [
        ("naive_bayes", {"var_smoothing": 0.1}),
        ("softmax", {"learning_rate": 0.1, "l2": 1e-3, "epochs": 5, "batch_size": 32, "seed": 1}),
        (
            "adaboost",
            {
                "n_estimators": 5,
                "max_depth": 2,
                "learning_rate": 1.0,
                "n_bins": 16,
                "max_features": 1.0,
                "seed": 1,
            },
        ),
    ],
)
def test_sklearn_baselines_run_under_same_protocol(model_key, params, tiny):
    res = run_baseline(model_key, params, tiny, repeats=1, log=lambda *_: None)
    assert set(res["splits"]) == {"train", "val", "test"}
    assert res["splits"]["test"]["accuracy"] > 0.3
    assert type(sklearn_counterpart(model_key, params, 600)).__name__ == res["sklearn_class"]


def test_tracker_modes(monkeypatch):
    assert resolve_mode("disabled") == "disabled" and resolve_mode("offline") == "offline"
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert resolve_mode("auto") == "disabled"
    t = Tracker("disabled")
    with t.run("x", config={"a": 1}) as run:
        run.log({"loss": 1.0}, step=1)
        run.summary({"acc": 1.0})
        run.log_figure("fig", None)
    assert t.runs == [("x", "disabled", None)]


def test_tracker_offline_writes_local_run(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    t = Tracker("offline", directory=tmp_path)
    with t.run("unit-test-run", group="g", job_type="test", config={"lr": 0.1}) as run:
        run.log({"loss": 0.5}, step=1)
        run.summary({"acc": 0.9})
    assert (tmp_path / "wandb").is_dir() and any((tmp_path / "wandb").glob("offline-run-*"))
