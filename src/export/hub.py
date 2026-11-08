"""Upload the bundle to the Hugging Face Hub (uses the logged-in CLI session or HF_TOKEN).

After an upload the commit hash of the Hub repository is written to ``web/model-lock.json``
(``--write-lock``): the website is built against *that* revision, never against ``main``, and the
lock records the bundle id so a mismatch between code, manifest and remote files is detectable
(``scripts``/CI compare the three).
"""

import json
from pathlib import Path

DEFAULT_REPO_NAME = "WriteRHand-MNIST"
LOCK_PATH = Path(__file__).resolve().parents[2] / "web" / "model-lock.json"


def write_lock(repo_id, revision, manifest, path=LOCK_PATH):
    lock = {
        "repo": repo_id,
        "revision": revision,
        "bundle_id": manifest["bundle_id"],
        "version": manifest["version"],
        "schema": manifest["schema"],
        "manifest_sha256": manifest["_manifest_sha256"],
    }
    Path(path).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def upload(artifacts_dir, repo_id=None, dry_run=False, write_lock_file=False, log=print):
    from huggingface_hub import HfApi

    from src.export.tracespec import sha256_file

    bundle = Path(artifacts_dir) / "bundle"
    if not (bundle / "manifest.json").is_file():
        raise SystemExit("bundle not found - run `python -m src export` first.")
    api = HfApi()
    if repo_id is None:
        repo_id = f"{api.whoami()['name']}/{DEFAULT_REPO_NAME}"  # needs authentication
    files = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file())
    log(f"target: https://huggingface.co/{repo_id}")
    log("files: " + ", ".join(files))
    if dry_run:
        log("dry run - nothing uploaded")
        return None
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    info = api.upload_folder(
        folder_path=str(bundle),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"WriteRHand bundle {manifest['bundle_id']} (schema {manifest['schema']})",
        # the pre-2.0 layout (single dataset at the repository root) is superseded
        delete_patterns=["models/*", "python/*", "test_vectors.json", "results.json"],
    )
    revision = getattr(info, "oid", None) or str(info).rsplit("/", 1)[-1]
    log(f"uploaded: {info}  revision={revision}")
    if write_lock_file:
        manifest["_manifest_sha256"] = sha256_file(bundle / "manifest.json")
        log("lock:", write_lock(repo_id, revision, manifest))
    return revision
