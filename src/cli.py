"""Command line interface:  ``uv run python -m src <command> [options]``

    prepare     download MNIST, build the stratified split + normalisation statistics
    tune        sequential hyper-parameter sweeps (train/validation only)
    train       train the selected configuration, evaluate train/val/test, save the model
    baselines   train the scikit-learn counterparts under the same protocol
    export      ONNX / JSON export + validation against the Python model
    augment     optional study: MLP with vs without data augmentation (3 seeds)
    report      tables and figures from results/
    upload      push the artifacts to the Hugging Face Hub

Typical session:  prepare -> tune --model all -> train --model all -> baselines -> report -> export
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.experiments.search import CLASS_NAME, SMOKE, SPECS, subset, tune
from src.experiments.tracking import Tracker
from src.utils.paths import ARTIFACTS_DIR, RESULTS_DIR

MODEL_CHOICES = [*SPECS, "all"]


def _models(arg):
    return list(SPECS) if arg == "all" else [arg]


def _load(smoke=False):
    from src.preprocessing.preprocess import load_dataset

    data, meta = load_dataset()
    if smoke:
        data = subset(data, SMOKE["n_train"], SMOKE["n_val"])
        data["X_test"], data["y_test"] = data["X_test"][:1000], data["y_test"][:1000]
    return data, meta


def cmd_prepare(args):
    data, meta = _load()
    print(json.dumps(meta, indent=2))
    print({k: v.shape for k, v in data.items()})


def cmd_tune(args):
    data, _ = _load()
    # the test split is deliberately not handed to the search
    data = {k: v for k, v in data.items() if not k.endswith("_test")}
    tracker = Tracker(args.wandb)
    for m in _models(args.model):
        tune(
            m,
            data,
            tracker=tracker,
            smoke=args.smoke,
            out_dir=Path(args.results_dir) / "tuning" if args.results_dir else None,
        )
    print("tracking mode:", tracker.mode, "| runs:", len(tracker.runs))


def _best_params(model_key, results_dir):
    path = Path(results_dir or RESULTS_DIR) / "tuning" / f"{CLASS_NAME[model_key]}.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found - run `tune --model {model_key}` first.")
    return json.loads(path.read_text(encoding="utf-8"))["best_params"]


def cmd_train(args):
    from src.experiments.final import train_final

    data, meta = _load(args.smoke)
    tracker = Tracker(args.wandb)
    rd = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    ad = Path(args.artifacts_dir) if args.artifacts_dir else ARTIFACTS_DIR
    for m in _models(args.model):
        params = _best_params(m, rd)
        train_final(
            m,
            params,
            data,
            meta,
            tracker=tracker,
            out_dir=rd / "final",
            art_dir=ad / CLASS_NAME[m],
            repeats=2 if args.smoke else 5,
        )
    print("tracking mode:", tracker.mode, "| runs:", len(tracker.runs))


def cmd_baselines(args):
    from src.experiments.final import run_baseline

    data, _ = _load(args.smoke)
    rd = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    out = rd / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    for m in _models(args.model):
        res = run_baseline(m, _best_params(m, rd), data, repeats=2 if args.smoke else 5)
        preds = res.pop("predictions")
        np.savez_compressed(out / f"sklearn_{CLASS_NAME[m]}_predictions.npz", **preds)
        (out / f"sklearn_{CLASS_NAME[m]}.json").write_text(
            json.dumps(
                res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
            ),
            encoding="utf-8",
        )


def cmd_augment(args):
    from src.experiments.augment import run_study, save_study

    data, _ = _load(args.smoke)
    rd = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    params = _best_params("mlp", rd)
    if args.smoke:
        params = {**params, "epochs": 3}
    study = run_study(data, params, seeds=(42,) if args.smoke else (42, 43, 44), k=args.copies)
    print("saved", save_study(study, rd / "augmentation.json"))


def cmd_report(args):
    from src.experiments.report import build_report

    build_report(Path(args.results_dir) if args.results_dir else RESULTS_DIR)


def cmd_export(args):
    from src.export.bundle import export_all

    export_all(
        Path(args.artifacts_dir) if args.artifacts_dir else ARTIFACTS_DIR,
        Path(args.results_dir) if args.results_dir else RESULTS_DIR,
    )


def cmd_upload(args):
    from src.export.hub import upload

    upload(
        Path(args.artifacts_dir) if args.artifacts_dir else ARTIFACTS_DIR,
        repo_id=args.repo_id,
        dry_run=args.dry_run,
    )


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m src",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, model=False, wandb=False, smoke=False):
        sp = sub.add_parser(name, help=fn.__doc__)
        sp.set_defaults(func=fn)
        sp.add_argument("--results-dir", default=None)
        sp.add_argument("--artifacts-dir", default=None)
        if model:
            sp.add_argument("--model", choices=MODEL_CHOICES, default="all")
        if wandb:
            sp.add_argument(
                "--wandb",
                choices=["auto", "online", "offline", "disabled"],
                default="auto",
                help="Weights & Biases mode (default: auto)",
            )
        if smoke:
            sp.add_argument("--smoke", action="store_true", help="tiny fast run (CI sanity check)")
        return sp

    add("prepare", cmd_prepare)
    add("tune", cmd_tune, model=True, wandb=True, smoke=True)
    add("train", cmd_train, model=True, wandb=True, smoke=True)
    add("baselines", cmd_baselines, model=True, smoke=True)
    aug = add("augment", cmd_augment, smoke=True)
    aug.add_argument("--copies", type=int, default=2, help="augmented copies per training image")
    add("report", cmd_report)
    add("export", cmd_export)
    up = add("upload", cmd_upload)
    up.add_argument("--repo-id", default=None, help="e.g. <user>/WriteRHand-MNIST")
    up.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
