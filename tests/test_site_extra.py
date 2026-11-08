"""Website data of the measured models that are not in the browser bundle (linear SVM, stacking)."""

import json

import numpy as np

from src.experiments.report import EXTRA_SCHEMA, build_extra_models, write_extra_models
from src.utils.paths import ROOT


def _split(acc):
    cm = np.diag([10] * 10).tolist()
    cm[4][9] = 3
    return {
        "accuracy": acc,
        "precision_macro": acc,
        "recall_macro": acc,
        "f1_macro": acc,
        "predict_time_sec": 0.1,
        "predict_time_ms_per_sample": 0.01,
        "report": "not copied",
        "confusion_matrix": cm,
    }


def _final(name, acc, fit, **extra):
    return {
        "model_name": name,
        "params": {"l2": 1e-4},
        "train_time_sec": fit,
        "latency_ms_single_sample": 0.5,
        "splits": {s: _split(acc) for s in ("train", "val", "test")},
        **extra,
    }


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_none_without_results(tmp_path):
    assert build_extra_models(tmp_path) is None


def test_svm_and_stacking_entries(tmp_path):
    fin = tmp_path / "final"
    _write(fin / "LinearSVM.json", _final("LinearSVM", 0.92, 10.0, history={"loss": [1.0, 0.5]}))
    _write(fin / "CNN.json", _final("CNN", 0.99, 100.0))
    _write(
        fin / "StackingEnsemble.json",
        _final(
            "StackingEnsemble",
            0.995,
            2.0,
            base_models=["CNN", "LinearSVM"],
            meta_fit_sec=2.0,
            base_test_accuracy={"CNN": 0.99, "LinearSVM": 0.92},
            block_importance={"CNN": 0.3, "LinearSVM": 0.1},
            cv_on_validation={
                "0.01": {"cv_val_acc_mean": 0.98, "cv_val_acc_sd": 0.01},
                "0.0001": {"cv_val_acc_mean": 0.99, "cv_val_acc_sd": 0.01},
            },
            selected_l2=1e-4,
        ),
    )
    _write(
        tmp_path / "tuning" / "LinearSVM.json",
        {
            "best_params": {"loss": "hinge"},
            "n_trials": 2,
            "trials": [
                {
                    "param": "loss",
                    "value": v,
                    "train_acc": 0.9,
                    "val_acc": a,
                    "train_time_sec": 1.0,
                    "stage": "sweep:loss",
                }
                for v, a in (("hinge", 0.91), ("squared_hinge", 0.89))
            ],
        },
    )
    out = build_extra_models(tmp_path)
    ids = [m["id"] for m in out["models"]]
    assert ids == ["svm", "stacking"]
    svm, stack = out["models"]
    assert svm["in_browser"] is False and stack["in_browser"] is False
    # a model that the browser bundle also has is marked as runnable there
    runnable = build_extra_models(tmp_path, ["svm", "svm_aug"])
    assert [m["in_browser"] for m in runnable["models"]] == [True, False]
    assert svm["splits"]["test"]["accuracy"] == 0.92
    assert "report" not in svm["splits"]["test"]
    assert svm["top_confusions"][0] == {"true": 4, "pred": 9, "count": 3}
    assert out["tuning"]["svm"]["sweeps"]["loss"][1]["val_acc"] == 0.89
    # cost of the stacking model: the base models it needs plus its meta-learner
    assert stack["train_time_sec"] == 110.0 + 2.0
    assert stack["stacking"]["fitted_on"] == "val"
    assert [b["id"] for b in stack["stacking"]["bases"]] == ["cnn", "svm"]
    # the l2 sweep is the 5-fold CV inside the validation split, sorted by value
    assert [p["value"] for p in out["tuning"]["stacking"]["sweeps"]["l2"]] == [1e-4, 0.01]


def test_write_is_deterministic(tmp_path):
    fin = tmp_path / "res" / "final"
    _write(fin / "LinearSVM.json", _final("LinearSVM", 0.92, 10.0))
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_extra_models({"mnist": tmp_path / "res", "other": tmp_path / "none"}, a)
    write_extra_models({"mnist": tmp_path / "res", "other": tmp_path / "none"}, b)
    assert a.read_bytes() == b.read_bytes()
    data = json.loads(a.read_text(encoding="utf-8"))
    assert data["schema"] == EXTRA_SCHEMA and list(data["datasets"]) == ["mnist"]


def test_committed_site_file_matches_results(tmp_path):
    """web/public/data/extra_models.json must be the output of ``python -m src site-extra`` on the committed results."""
    committed = ROOT / "web" / "public" / "data" / "extra_models.json"
    fresh = tmp_path / "extra.json"
    manifest = json.loads(
        (ROOT / "web" / "public" / "data" / "manifest.json").read_text(encoding="utf-8")
    )
    write_extra_models({"mnist": ROOT / "results"}, fresh, manifest)
    assert json.loads(committed.read_text(encoding="utf-8")) == json.loads(
        fresh.read_text(encoding="utf-8")
    )


def _pair_row(acc, aug):
    return {
        "class": "X",
        "train_acc": acc,
        "val_acc": acc,
        "test_acc": acc,
        "test_f1_macro": acc,
        "fit_time_sec": 1.0,
        "latency_ms": 0.1,
        "augmentation": {
            "ops": ["rotation", "translation"],
            "k": 2,
            "seed": 1,
            "ranges": {"rotation": 30.0},
        }
        if aug
        else None,
        "test_rotation_acc": {"0": acc, "30": acc - 0.1},
        "test_stress_acc": {"rotation": acc},
    }


def test_pairs_of_plain_and_augmented_models(tmp_path):
    _write(
        tmp_path / "augmentation_pairs.json",
        {
            "schema": 1,
            "rotation_angles": [0, 30],
            "models": {
                "mlp": {"plain": _pair_row(0.98, False), "augmented": _pair_row(0.984, True)},
                "softmax": {"plain": _pair_row(0.92, False)},
            },
        },
    )
    out = build_extra_models(tmp_path)
    assert "models" not in out
    pairs = out["pairs"]
    assert list(pairs["models"]) == ["mlp"]  # an incomplete pair is dropped
    assert pairs["models"]["mlp"]["augmented"]["test_rotation_acc"]["30"] == 0.984 - 0.1
    assert "class" not in pairs["models"]["mlp"]["plain"]
    # the recipe completes the record with the default range of every listed operation
    assert pairs["recipe"] == {
        "ops": ["rotation", "translation"],
        "copies": 2,
        "ranges": {"rotation": 30.0, "translation": 2.0},
    }
    assert pairs["rotation_angles"] == [0, 30]
