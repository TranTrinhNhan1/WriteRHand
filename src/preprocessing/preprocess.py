"""MNIST loading, splitting and normalisation.

Protocol (version ``PREPROCESS_VERSION``):

* the official 60 000 training images are split into train / validation with a
  *stratified*, seeded split (default 50 000 / 10 000); the official 10 000 test
  images are never touched until the final evaluation;
* pixels are scaled to [0, 1], flattened to 784 and standardised with a single
  scalar mean / std computed on the *training portion only*;
* the same ``(mean, std)`` pair is shipped with every exported model so that the
  browser demo applies exactly the same transform.
"""

import hashlib
import json

import numpy as np
import torch
from torchvision.datasets import MNIST

from src.utils.paths import DATA_DIR, PROCESSED_PATH, SEED

PREPROCESS_VERSION = "mnist-global-standardize-v2-stratified"
N_TRAIN = 50_000
N_FEATURES = 784
N_CLASSES = 10


def load_mnist(root=DATA_DIR):
    """Return ``(X_train_raw, y_train, X_test_raw, y_test)`` as torch tensors (uint8 images)."""
    try:
        train = MNIST(root=str(root), train=True, download=True)
        test = MNIST(root=str(root), train=False, download=True)
    except Exception as e:  # network / disk problems
        raise RuntimeError(f"Cannot load MNIST from {root}: {type(e).__name__}: {e}") from e
    return train.data, train.targets, test.data, test.targets


def flatten(X_raw):
    """uint8 (N, 28, 28) -> float32 (N, 784) scaled to [0, 1]."""
    return X_raw.reshape(-1, N_FEATURES).float() / 255.0


def standardize(X, mean, std):
    return (X - mean) / (std + 1e-8)


def stratified_split_indices(y, n_val, seed=SEED):
    """Seeded stratified split. Returns ``(train_idx, val_idx)`` (sorted numpy int64 arrays).

    Each class contributes ``round(n_val * class_share)`` validation samples (largest-remainder
    rounding so the total is exactly ``n_val``).
    """
    y = np.asarray(y).astype(np.int64)
    n = len(y)
    if not 0 < n_val < n:
        raise ValueError("n_val must satisfy 0 < n_val < len(y)")
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    exact = counts * (n_val / n)
    n_per_class = np.floor(exact).astype(int)
    remainder = n_val - n_per_class.sum()
    order = np.argsort(-(exact - n_per_class), kind="stable")
    n_per_class[order[:remainder]] += 1

    val_parts = []
    for c, k in zip(classes, n_per_class, strict=True):
        idx = np.flatnonzero(y == c)
        val_parts.append(rng.permutation(idx)[:k])
    val_idx = np.sort(np.concatenate(val_parts))
    mask = np.ones(n, dtype=bool)
    mask[val_idx] = False
    return np.flatnonzero(mask), val_idx


def split_fingerprint(train_idx, val_idx):
    """Short hash identifying a train/val split (logged with every experiment)."""
    h = hashlib.sha256()
    h.update(np.asarray(train_idx, dtype=np.int64).tobytes())
    h.update(np.asarray(val_idx, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def prepare_mnist(Xtr_raw, ytr_raw, Xte_raw, yte_raw, has_validation=True,
                  seed=SEED, n_train=N_TRAIN, stratified=True):
    """Split + normalise.

    ``has_validation=True``  -> ``X_train, y_train, X_val, y_val, X_test, y_test, mean, std``
    ``has_validation=False`` -> ``X_train, y_train, X_test, y_test, mean, std`` (used inside
    K-fold: the statistics come from the fold's training part only, so nothing leaks).
    """
    if has_validation:
        n_val = len(Xtr_raw) - n_train
        if stratified:
            tr_idx, va_idx = stratified_split_indices(ytr_raw, n_val, seed)
        else:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(Xtr_raw), generator=g).numpy()
            tr_idx, va_idx = np.sort(perm[:n_train]), np.sort(perm[n_train:])
        tr_idx, va_idx = torch.from_numpy(tr_idx), torch.from_numpy(va_idx)

        X_scaled_train = flatten(Xtr_raw[tr_idx])
        mean = X_scaled_train.mean().item()
        std = X_scaled_train.std().item()

        X_train = standardize(X_scaled_train, mean, std)
        X_val = standardize(flatten(Xtr_raw[va_idx]), mean, std)
        X_test = standardize(flatten(Xte_raw), mean, std)
        return X_train, ytr_raw[tr_idx], X_val, ytr_raw[va_idx], X_test, yte_raw, mean, std

    X_scaled_train = flatten(Xtr_raw)
    mean = X_scaled_train.mean().item()
    std = X_scaled_train.std().item()
    X_train = standardize(X_scaled_train, mean, std)
    X_test = standardize(flatten(Xte_raw), mean, std)
    return X_train, ytr_raw, X_test, yte_raw, mean, std


def build_dataset(root=DATA_DIR, seed=SEED, n_train=N_TRAIN):
    """Full pipeline -> dict of numpy arrays (float32 features, int64 labels) + metadata."""
    Xtr_raw, ytr_raw, Xte_raw, yte_raw = load_mnist(root)
    n_val = len(Xtr_raw) - n_train
    tr_idx, va_idx = stratified_split_indices(ytr_raw, n_val, seed)
    X_all = flatten(Xtr_raw).numpy()
    X_test_raw = flatten(Xte_raw).numpy()
    mean = float(X_all[tr_idx].mean(dtype=np.float64))
    std = float(X_all[tr_idx].std(dtype=np.float64, ddof=1))

    def norm(a):
        return ((a - mean) / (std + 1e-8)).astype(np.float32)

    meta = {
        "preprocess_version": PREPROCESS_VERSION,
        "seed": seed,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(yte_raw)),
        "mean": mean,
        "std": std,
        "split_fingerprint": split_fingerprint(tr_idx, va_idx),
    }
    data = {
        "X_train": norm(X_all[tr_idx]), "y_train": ytr_raw.numpy()[tr_idx].astype(np.int64),
        "X_val": norm(X_all[va_idx]), "y_val": ytr_raw.numpy()[va_idx].astype(np.int64),
        "X_test": norm(X_test_raw), "y_test": yte_raw.numpy().astype(np.int64),
    }
    return data, meta


def save_processed_data(root=PROCESSED_PATH, seed=SEED, n_train=N_TRAIN):
    root.mkdir(parents=True, exist_ok=True)
    data, meta = build_dataset(seed=seed, n_train=n_train)
    np.savez_compressed(root / "mnist.npz", **data)
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_dataset(root=PROCESSED_PATH, seed=SEED, n_train=N_TRAIN):
    """Load (building and caching if necessary) the processed split. Returns ``(data, meta)``."""
    npz, meta_path = root / "mnist.npz", root / "meta.json"
    meta = None
    if npz.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("preprocess_version") != PREPROCESS_VERSION
                or meta.get("seed") != seed or meta.get("n_train") != n_train):
            meta = None  # stale cache
    if meta is None:
        meta = save_processed_data(root, seed, n_train)
    with np.load(npz) as f:
        data = {k: f[k] for k in f.files}
    return data, meta


def load_processed_data(root=PROCESSED_PATH):
    """Backward-compatible tuple of torch tensors (X_train, y_train, X_val, y_val, X_test, y_test)."""
    data, _ = load_dataset(root)
    return tuple(torch.from_numpy(data[k]) for k in
                 ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"))
