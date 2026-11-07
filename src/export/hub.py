"""Upload the bundle to the Hugging Face Hub (uses the logged-in CLI session or HF_TOKEN)."""

from pathlib import Path

DEFAULT_REPO_NAME = "WriteRHand-MNIST"


def upload(artifacts_dir, repo_id=None, dry_run=False, log=print):
    from huggingface_hub import HfApi

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
    info = api.upload_folder(
        folder_path=str(bundle),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload WriteRHand-MNIST bundle",
    )
    log(f"uploaded: {info}")
    return repo_id
