"""Weights & Biases tracking with graceful fallbacks.

Modes:  ``online`` | ``offline`` | ``disabled`` | ``auto`` (default).
``auto`` -> online when a W&B API key is available (``WANDB_API_KEY`` or a stored login),
otherwise offline (runs are written to ``./wandb`` and can be uploaded later with
``wandb sync``). If the ``wandb`` package is missing, tracking is disabled. The project's own JSON
result files are always written, so nothing depends on W&B being reachable.
"""

import os
from contextlib import contextmanager

PROJECT = "WriteRHand"
os.environ.setdefault("WANDB_SILENT", "true")  # keep the CLI output readable


def _has_stored_key():
    """True when ``wandb login`` stored a key in ~/.netrc (no network, no prompt)."""
    import netrc

    try:
        return netrc.netrc().authenticators("api.wandb.ai") is not None
    except (OSError, netrc.NetrcParseError):
        return False


def resolve_mode(mode="auto"):
    if mode != "auto":
        return mode
    env = os.environ.get("WANDB_MODE")
    if env in ("online", "offline", "disabled"):
        return env
    try:
        import wandb  # noqa: F401
    except ImportError:
        return "disabled"
    return "online" if (os.environ.get("WANDB_API_KEY") or _has_stored_key()) else "offline"


class NullRun:
    """Stand-in used when tracking is disabled; keeps the same interface."""

    mode = "disabled"
    url = None

    def log(self, data, step=None):
        pass

    def summary(self, data):
        pass

    def log_figure(self, key, fig):
        pass

    def log_artifact(self, path, name, type_="model", metadata=None):
        pass


class WandbRun(NullRun):
    def __init__(self, run, mode):
        self._run = run
        self.mode = mode
        self.url = getattr(run, "url", None) if mode == "online" else None

    def log(self, data, step=None):
        self._run.log(data, step=step)

    def summary(self, data):
        for k, v in data.items():
            self._run.summary[k] = v

    def log_figure(self, key, fig):
        import wandb

        self._run.log({key: wandb.Image(fig)})

    def log_artifact(self, path, name, type_="model", metadata=None):
        import wandb

        art = wandb.Artifact(name, type=type_, metadata=metadata or {})
        art.add_file(str(path))
        self._run.log_artifact(art)


class Tracker:
    def __init__(self, mode="auto", project=PROJECT, entity=None, directory=None):
        self.mode = resolve_mode(mode)
        self.project = project
        self.entity = entity or os.environ.get("WANDB_ENTITY")
        if directory is None:
            from src.utils.paths import ROOT

            directory = ROOT
        self.directory = str(directory)
        self.runs = []  # (name, mode, url) of every run opened

    @contextmanager
    def run(self, name, *, group=None, job_type=None, config=None, tags=None):
        if self.mode == "disabled":
            run = NullRun()
            yield run
            self.runs.append((name, "disabled", None))
            return
        import wandb

        r = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            group=group,
            job_type=job_type,
            config=config or {},
            tags=tags or [],
            mode=self.mode,
            dir=self.directory,
            reinit=True,
        )
        wrapped = WandbRun(r, self.mode)
        try:
            yield wrapped
        finally:
            r.finish()
            self.runs.append((name, self.mode, wrapped.url))
