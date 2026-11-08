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

from src.utils.paths import DATA_DIR, PROCESSED_PATH, SEED

PREPROCESS_VERSION = "mnist-global-standardize-v2-stratified"
N_TRAIN = 50_000
N_FEATURES = 784
N_CLASSES = 10


# ------------------------------------------------------------------ datasets
# Every dataset is 28x28 grayscale with 10 classes, so every model keeps the same input and output
# size; what differs is the *meaning* of the 10 labels (``class_names``) and the pool of images.
DATASETS = {
    "mnist": {
        "title": "MNIST",
        "short": "MNIST",
        "classes": [str(i) for i in range(10)],
        "kind": "digits",
        "source": "LeCun et al. (1998); torchvision.datasets.MNIST",
    },
    "fashion_mnist": {
        "title": "Fashion-MNIST",
        "short": "Fashion",
        "classes": [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
        "kind": "clothing",
        "source": "Xiao et al. (2017); torchvision.datasets.FashionMNIST",
    },
    "emnist_digits": {
        "title": "EMNIST Digits",
        "short": "EMNIST",
        "classes": [str(i) for i in range(10)],
        "kind": "digits",
        "source": "Cohen et al. (2017); torchvision.datasets.EMNIST(split='digits')",
    },
}
DEFAULT_DATASET = "mnist"
N_VAL = 10_000


def check_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose one of {list(DATASETS)}.")
    return name


def load_raw(name=DEFAULT_DATASET, root=DATA_DIR):
    """Return ``(X_train_raw, y_train, X_test_raw, y_test)`` as torch tensors (uint8 images)."""
    check_dataset(name)
    try:
        if name == "mnist":
            from torchvision.datasets import MNIST as DS

            train, test = (
                DS(str(root), train=True, download=True),
                DS(str(root), train=False, download=True),
            )
            return train.data, train.targets, test.data, test.targets
        if name == "fashion_mnist":
            from torchvision.datasets import FashionMNIST as DS

            train, test = (
                DS(str(root), train=True, download=True),
                DS(str(root), train=False, download=True),
            )
            return train.data, train.targets, test.data, test.targets
        from torchvision.datasets import EMNIST

        train = EMNIST(str(root), split="digits", train=True, download=True)
        test = EMNIST(str(root), split="digits", train=False, download=True)
        # EMNIST images are stored transposed relative to MNIST; undo it so digits are upright
        return (
            train.data.transpose(1, 2).contiguous(),
            train.targets,
            test.data.transpose(1, 2).contiguous(),
            test.targets,
        )
    except Exception as e:  # network / disk problems
        raise RuntimeError(f"Cannot load {name} from {root}: {type(e).__name__}: {e}") from e


def load_mnist(root=DATA_DIR):
    """Backward-compatible alias: MNIST only."""
    return load_raw("mnist", root)


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


def prepare_mnist(
    Xtr_raw,
    ytr_raw,
    Xte_raw,
    yte_raw,
    has_validation=True,
    seed=SEED,
    n_train=N_TRAIN,
    stratified=True,
):
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


def build_dataset(root=DATA_DIR, seed=SEED, n_train=N_TRAIN, name=DEFAULT_DATASET, n_val=N_VAL):
    """Full pipeline -> dict of numpy arrays (float32 features, int64 labels) + metadata.

    The official training set is split (stratified, seeded) into ``n_train`` / ``n_val`` images; if
    the official set is larger than ``n_train + n_val`` (EMNIST Digits has 240 000) a stratified
    pool of that size is drawn first, so every dataset is trained with the same budget. The
    official test set is used whole and only for final reporting.
    """
    Xtr_raw, ytr_raw, Xte_raw, yte_raw = load_raw(name, root)
    pool = None
    if len(Xtr_raw) > n_train + n_val:
        _, pool = stratified_split_indices(ytr_raw, n_train + n_val, seed)
        Xtr_raw, ytr_raw = Xtr_raw[torch.from_numpy(pool)], ytr_raw[torch.from_numpy(pool)]
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
        **(
            {}
            if name == DEFAULT_DATASET
            else {"dataset": name, "pool_fingerprint": split_fingerprint(pool, pool)}
            if pool is not None
            else {"dataset": name}
        ),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(yte_raw)),
        "mean": mean,
        "std": std,
        "split_fingerprint": split_fingerprint(tr_idx, va_idx),
    }
    data = {
        "X_train": norm(X_all[tr_idx]),
        "y_train": ytr_raw.numpy()[tr_idx].astype(np.int64),
        "X_val": norm(X_all[va_idx]),
        "y_val": ytr_raw.numpy()[va_idx].astype(np.int64),
        "X_test": norm(X_test_raw),
        "y_test": yte_raw.numpy().astype(np.int64),
    }
    return data, meta


def _cache_paths(root, name):
    """MNIST keeps its historical file names; other datasets live in their own sub-directory."""
    if name == DEFAULT_DATASET:
        return root / "mnist.npz", root / "meta.json"
    return root / name / "data.npz", root / name / "meta.json"


def save_processed_data(root=PROCESSED_PATH, seed=SEED, n_train=N_TRAIN, name=DEFAULT_DATASET):
    npz, meta_path = _cache_paths(root, name)
    npz.parent.mkdir(parents=True, exist_ok=True)
    data, meta = build_dataset(seed=seed, n_train=n_train, name=name)
    np.savez_compressed(npz, **data)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_dataset(root=PROCESSED_PATH, seed=SEED, n_train=N_TRAIN, name=DEFAULT_DATASET):
    """Load (building and caching if necessary) the processed split. Returns ``(data, meta)``."""
    check_dataset(name)
    npz, meta_path = _cache_paths(root, name)
    meta = None
    if npz.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("preprocess_version") != PREPROCESS_VERSION
            or meta.get("seed") != seed
            or meta.get("n_train") != n_train
        ):
            meta = None  # stale cache
    if meta is None:
        meta = save_processed_data(root, seed, n_train, name)
    with np.load(npz) as f:
        data = {k: f[k] for k in f.files}
    meta.setdefault("dataset", name)
    return data, meta


def load_processed_data(root=PROCESSED_PATH):
    """Backward-compatible tuple of torch tensors (X_train, y_train, X_val, y_val, X_test, y_test)."""
    data, _ = load_dataset(root)
    return tuple(
        torch.from_numpy(data[k])
        for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
    )
