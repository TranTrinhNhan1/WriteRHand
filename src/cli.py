"""Command line interface:  ``uv run python -m src <command> [options]``

    prepare     download MNIST, build the stratified split + normalisation statistics
    tune        sequential hyper-parameter sweeps (train/validation only)
    train       train the selected configuration, evaluate train/val/test, save the model
    baselines   train the scikit-learn counterparts under the same protocol
    stack       stacking ensemble over the trained base models (hold-out = validation split)
    export      ONNX / JSON export + validation against the Python model
    site-extra  web/public/data/extra_models.json (linear SVM, stacking: measured, not in the browser bundle)
    augment     optional study: MLP with vs without data augmentation (3 seeds)
    augment-pairs   plain vs augmented variant of every model (clean + robustness)
    augment-select  choose the augmentation of the deployed MLP / CNN on the validation split
    replay      record training replays (real checkpoints + metrics) for the explorer
    report      tables and figures from results/
    upload      push the artifacts to the Hugging Face Hub

Typical session:  prepare -> tune --model all -> train --model all -> baselines -> report -> export
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.experiments.search import BASELINE_MODELS, CLASS_NAME, SMOKE, SPECS, subset, tune
from src.experiments.tracking import Tracker
from src.preprocessing.preprocess import DATASETS
from src.utils.paths import dataset_dirs

MODEL_CHOICES = [*SPECS, "all"]


def _models(arg):
    return list(SPECS) if arg == "all" else [arg]


def _dataset(args):
    return getattr(args, "dataset", "mnist") or "mnist"


def _dirs(args):
    """Results / artifacts directories: explicit flags win, otherwise per-dataset defaults."""
    rd, ad = dataset_dirs(_dataset(args))
    return (
        Path(args.results_dir) if args.results_dir else rd,
        Path(args.artifacts_dir) if args.artifacts_dir else ad,
    )


def _load(smoke=False, dataset="mnist"):
    from src.preprocessing.preprocess import load_dataset

    data, meta = load_dataset(name=dataset)
    if smoke:
        data = subset(data, SMOKE["n_train"], SMOKE["n_val"])
        data["X_test"], data["y_test"] = data["X_test"][:1000], data["y_test"][:1000]
    return data, meta


def cmd_prepare(args):
    data, meta = _load(dataset=_dataset(args))
    print(json.dumps(meta, indent=2))
    print({k: v.shape for k, v in data.items()})


def cmd_tune(args):
    data, _ = _load(dataset=_dataset(args))
    rd, _ad = _dirs(args)
    # the test split is deliberately not handed to the search
    data = {k: v for k, v in data.items() if not k.endswith("_test")}
    tracker = Tracker(args.wandb, dataset=_dataset(args))
    for m in _models(args.model):
        tune(
            m,
            data,
            tracker=tracker,
            smoke=args.smoke,
            out_dir=rd / "tuning",
            light=_dataset(args) != "mnist" and args.light,
        )
    print("tracking mode:", tracker.mode, "| runs:", len(tracker.runs))


def _best_params(model_key, results_dir):
    path = Path(results_dir) / "tuning" / f"{CLASS_NAME[model_key]}.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found - run `tune --model {model_key}` first.")
    return json.loads(path.read_text(encoding="utf-8"))["best_params"]


AUGMENT_RECIPE_FROM = "mlp"  # one recipe for every augmented variant: the one chosen for the MLP


def _augment_spec(model_key, rd):
    """Augmentation recipe of the ``<Model>_aug`` variants (chosen on the MLP's validation split).

    The same recipe is applied to every model so that each pair (plain, augmented) is a fair comparison;
    ``results/augmentation_deploy_<model>.json`` keeps the per-model selection runs.
    """
    path = Path(rd) / f"augmentation_deploy_{AUGMENT_RECIPE_FROM}.json"
    if not path.is_file():
        raise SystemExit(
            f"{path} not found - run `augment-select --model {AUGMENT_RECIPE_FROM}` first."
        )
    spec = json.loads(path.read_text(encoding="utf-8"))["spec"]
    if spec is None:
        raise SystemExit(f"{path}: no candidate satisfied the selection rule.")
    return spec


def cmd_train(args):
    from src.experiments.final import train_final

    data, meta = _load(args.smoke, _dataset(args))
    tracker = Tracker(args.wandb, dataset=_dataset(args))
    rd, ad = _dirs(args)
    for m in _models(args.model):
        params = _best_params(m, rd)
        train_final(
            m,
            params,
            data,
            meta,
            tracker=tracker,
            out_dir=rd / "final",
            art_dir=ad / (CLASS_NAME[m] + ("_aug" if args.augment else "")),
            repeats=2 if args.smoke else 5,
            augment=_augment_spec(m, rd) if args.augment else None,
        )
    print("tracking mode:", tracker.mode, "| runs:", len(tracker.runs))


def _baseline_models(arg):
    return list(BASELINE_MODELS) if arg == "all" else [arg]


def cmd_baselines(args):
    from src.experiments.final import run_baseline

    data, _ = _load(args.smoke, _dataset(args))
    rd, _ad = _dirs(args)
    out = rd / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    for m in _baseline_models(args.model):
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
    from src.experiments.augment import CONDITIONS, run_study_cli

    data, _ = _load(args.smoke, _dataset(args))
    rd, _ad = _dirs(args)
    params = _best_params(args.model, rd)
    conds = CONDITIONS
    seeds = tuple(range(42, 42 + args.seeds))
    if args.smoke:
        params = {**params, "epochs": 2}
        seeds = (42,)
        conds = {k: CONDITIONS[k] for k in ("baseline", "rotation", "salt_pepper")}
    print("saved", run_study_cli(data, args.model, params, seeds, args.copies, conds, rd))


def cmd_augment_select(args):
    """Choose the rotation range of an augmented model variant on the validation split."""
    from src.experiments.augment import select_deployed
    from src.models.base import BaseClassifier

    data, _ = _load(args.smoke, _dataset(args))
    rd, ad = _dirs(args)
    params = _best_params(args.model, rd)
    if args.smoke:
        params = {**params, "epochs": 2}
    base = BaseClassifier.load(ad / CLASS_NAME[args.model] / "model.npz")
    kw = {"rotations": tuple(float(r) for r in args.rotations)} if args.rotations else {}
    res = select_deployed(data, args.model, params, base, **kw)
    out = rd / f"augmentation_deploy_{args.model}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("chosen:", res["chosen"], "->", out)


def cmd_augment_pairs(args):
    """Plain vs augmented variant of every model: clean and robustness accuracy on the test split."""
    from src.experiments.augment_pairs import run_pairs

    data, _ = _load(dataset=_dataset(args))
    rd, ad = _dirs(args)
    run_pairs(data, art_root=ad, res_root=rd)
    print("wrote", rd / "augmentation_pairs.json")


def cmd_replay(args):
    """Record training replays (checkpoints + metrics of the real training run) for the explorer."""
    from src.experiments.final import load_json
    from src.experiments.replay import adaboost_replay, record_replay, save_replay
    from src.models.base import BaseClassifier

    data, meta = _load(dataset=_dataset(args))
    rd, ad = _dirs(args)
    for m in (
        _baseline_models(args.model)
        if args.model != "all"
        else ["softmax", "mlp", "cnn", "adaboost"]
    ):
        if m == "naive_bayes":
            print("naive_bayes: closed-form fit, no iterative training to replay")
            continue
        cname = CLASS_NAME[m] + ("_aug" if args.augment else "")
        params = _best_params(m, rd)
        path = ad / cname / "model.npz"
        fit_xy = None
        if args.augment:
            from src.experiments.augment import apply_spec

            fit_xy = apply_spec(data["X_train"], data["y_train"], _augment_spec(m, rd))
        if m == "adaboost":
            fin = load_json(rd / "final" / f"{cname}.json")
            rep = adaboost_replay(
                BaseClassifier.load(path), fin["history"], data, meta, _dataset(args), params
            )
        else:
            rep = record_replay(
                m, SPECS[m]["cls"], params, data, meta, path, _dataset(args), fit_xy=fit_xy
            )
        out = save_replay(rep, ad / "replay" / f"{cname}.json")
        print(
            f"{m}: {len(rep['snapshots'])} snapshots, final_matches_deployed={rep['final_matches_deployed']} -> {out}"
        )


def cmd_stack(args):
    """Stacking ensemble over the trained base models (meta-learner fitted on the validation split)."""
    from src.experiments.stacking import run_stacking

    data, meta = _load(args.smoke, _dataset(args))
    rd, ad = _dirs(args)
    run_stacking(data, meta, art_root=ad, out_dir=rd / "final", repeats=2 if args.smoke else 5)


def cmd_compare(args):
    """CNN vs MLP table (robustness, size, speed) and the cross-dataset evaluation."""
    from src.experiments.compare import cnn_vs_mlp, cross_eval
    from src.utils.paths import dataset_dirs as dd

    rd, ad = _dirs(args)
    ds = _dataset(args)
    res = cnn_vs_mlp(ds, rd, ad)
    if res:
        (rd / "cnn_vs_mlp.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("wrote", rd / "cnn_vs_mlp.json", "McNemar", res["mcnemar"])
    if ds == "emnist_digits":
        out = [cross_eval("mnist", ds, dd("mnist")[1])]
        (rd / "cross_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote", rd / "cross_eval.json", out[0]["rows"])


def cmd_report(args):
    from src.experiments.report import build_report

    rd, ad = _dirs(args)
    build_report(rd, ad, dataset=_dataset(args))


def cmd_export(args):
    """Export every dataset that has trained models (or one with --dataset)."""
    from src.export.bundle import export_all
    from src.utils.paths import ARTIFACTS_DIR

    names = [args.dataset] if args.dataset else list(DATASETS)
    sources = {}
    for n in names:
        rd, ad = dataset_dirs(n)
        if n == "mnist":
            rd = Path(args.results_dir) if args.results_dir else rd
            ad = Path(args.artifacts_dir) if args.artifacts_dir else ad
        if any((ad / c / "model.npz").is_file() for c in CLASS_NAME.values()):
            sources[n] = (ad, rd)
    root = Path(args.artifacts_dir) if args.artifacts_dir else ARTIFACTS_DIR
    export_all(sources, root)
    if args.site:
        _sync_site(root / "bundle")


def _sync_site(bundle):
    """The site commits only the manifest and the measured results; models stay in the bundle (HF)."""
    import json
    import shutil

    from src.utils.paths import ROOT

    out = ROOT / "web" / "public" / "data"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    man = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    shutil.copy(bundle / "manifest.json", out / "manifest.json")
    for name in man["datasets"]:
        (out / name).mkdir()
        shutil.copy(bundle / name / "results.json", out / name / "results.json")
    _write_site_extra(list(man["datasets"]))
    print(f"site data -> {out} ({len(man['datasets'])} datasets)")


def _write_site_extra(names):
    """web/public/data/extra_models.json: measured models that are not in the browser bundle."""
    from src.experiments.report import write_extra_models
    from src.utils.paths import ROOT

    path = ROOT / "web" / "public" / "data" / "extra_models.json"
    out = write_extra_models({n: dataset_dirs(n)[0] for n in names}, path)
    print(f"extra models -> {path} ({', '.join(out['datasets']) or 'none'})")


def cmd_site_extra(args):
    """Rewrite web/public/data/extra_models.json (linear SVM, stacking) from results/ only."""
    _write_site_extra([args.dataset] if args.dataset else list(DATASETS))


def cmd_upload(args):
    from src.export.hub import upload
    from src.utils.paths import ARTIFACTS_DIR

    upload(
        Path(args.artifacts_dir) if args.artifacts_dir else ARTIFACTS_DIR,
        repo_id=args.repo_id,
        dry_run=args.dry_run,
        write_lock_file=args.write_lock,
    )


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m src",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, model=False, wandb=False, smoke=False, dataset=True):
        sp = sub.add_parser(name, help=fn.__doc__)
        sp.set_defaults(func=fn)
        sp.add_argument("--results-dir", default=None)
        sp.add_argument("--artifacts-dir", default=None)
        if model:
            sp.add_argument("--model", choices=MODEL_CHOICES, default="all")
        if dataset:
            sp.add_argument(
                "--dataset",
                choices=list(DATASETS),
                default="mnist",
                help="dataset (default: mnist); results go to results/datasets/<name> unless overridden",
            )
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
    tn = add("tune", cmd_tune, model=True, wandb=True, smoke=True)
    tn.add_argument("--light", action="store_true", help="reduced sweeps (extra datasets only)")
    tr = add("train", cmd_train, model=True, wandb=True, smoke=True)
    tr.add_argument(
        "--augment",
        action="store_true",
        help="train the augmented variant (<Model>_aug) with the spec chosen by `augment-select`",
    )
    add("baselines", cmd_baselines, model=True, smoke=True)
    aug = add("augment", cmd_augment, smoke=True)
    aug.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    aug.add_argument("--seeds", type=int, default=3, help="number of training seeds per condition")
    aug.add_argument("--copies", type=int, default=1, help="augmented copies per training image")
    asel = add("augment-select", cmd_augment_select, smoke=True)
    asel.add_argument("--model", choices=list(SPECS), default="mlp")
    asel.add_argument(
        "--rotations",
        nargs="+",
        type=float,
        default=None,
        help="candidate rotation ranges in degrees (default: 15 30 45)",
    )
    add("augment-pairs", cmd_augment_pairs)
    rp = add("replay", cmd_replay, model=True)
    rp.add_argument(
        "--augment", action="store_true", help="replay the augmented training run (mlp / cnn)"
    )
    add("stack", cmd_stack, smoke=True)
    add("compare", cmd_compare)
    add("report", cmd_report)
    ex = add("export", cmd_export, dataset=False)
    ex.add_argument("--dataset", choices=list(DATASETS), default=None)
    ex.add_argument(
        "--site",
        action="store_true",
        help="also copy manifest.json and <dataset>/results.json into web/public/data (committed)",
    )
    se = add("site-extra", cmd_site_extra, dataset=False)
    se.add_argument("--dataset", choices=list(DATASETS), default=None)
    up = add("upload", cmd_upload, dataset=False)
    up.add_argument("--repo-id", default=None, help="e.g. <user>/WriteRHand-MNIST")
    up.add_argument("--dry-run", action="store_true")
    up.add_argument("--write-lock", action="store_true", help="write web/model-lock.json")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
