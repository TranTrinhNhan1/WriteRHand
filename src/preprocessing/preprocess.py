from pathlib import Path
from torchvision.datasets import MNIST
import torch

current_dir = Path.cwd()
PROJECT_ROOT = current_dir 
RAW_DIR = PROJECT_ROOT / "data"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH = RAW_DIR / "processed"

SEED = 42
torch.manual_seed(SEED)

def load_mnist(root=RAW_DIR):
    '''
    tra ve X_tr, y_tr, X_te, y_te
    '''
    try:
        train = MNIST(
            root=str(root),
            train=True,
            download=True
        )
        test = MNIST(
            root=str(root),
            train=False,
            download=True 
        )
        return train.data, train.targets, test.data, test.targets
    except Exception as e:
        print("Nguon mac dinh cua torchvision loi (", type(e).__name__, "), khong the tai du lieu")
        return None

def flatten(X_raw):
    return X_raw.reshape(-1, 784).float() / 255.0

def standardize(X, mean, std):
    return (X - mean) / (std + 1e-8)

def prepare_mnist(Xtr_raw, ytr_raw, Xte_raw, yte_raw, has_validation=True,
                  seed=SEED, n_train=50000):
    '''
    preprocess va chia tap train - validation - test
    dung cho K-fold thi has_validation=False
    '''
    
    if has_validation:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(Xtr_raw), generator=g) # tensor cac idx hoan vi ngau nhien
    
        tr_idx = perm[:n_train]
        va_idx = perm[n_train:]
        # Tính mean và std chỉ từ tập train
        X_raw_train = Xtr_raw[tr_idx]
        X_scaled_train = flatten(X_raw_train)

        mean = X_scaled_train.mean().item()
        std = X_scaled_train.std().item()
    
        X_train = standardize(X_scaled_train, mean, std)
        X_val = standardize(flatten(Xtr_raw[tr_idx]), mean, std)
        X_test = standardize(flatten(Xtr_raw[tr_idx]), mean, std)
    
        y_train = ytr_raw[tr_idx]
        y_val = ytr_raw[va_idx]
        y_test = yte_raw
    
        return X_train, y_train, X_val, y_val, X_test, y_test, mean, std
    else:
        X_scaled_train = flatten(Xtr_raw)
        mean = X_scaled_train.mean().item()
        std = X_scaled_train.std().item()
        
        X_train = standardize(X_scaled_train, mean, std)
        X_test = standardize(flatten(Xte_raw), mean, std)
        return X_train, ytr_raw, X_test, yte_raw, mean, std

def save_processed_data(root=PROCESSED_PATH):
    root.mkdir(parents=True, exist_ok=True)

    Xtr_raw, ytr_raw, Xte_raw, yte_raw = load_mnist()
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = prepare_mnist(Xtr_raw, ytr_raw, Xte_raw, yte_raw)

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

    torch.save(data, root / "mnist.pt")

def load_processed_data(root=PROCESSED_PATH):
    dir = root / "mnist.pt"
    if not dir.is_file():
        save_processed_data()

    data = torch.load(
        dir,
        map_location="cpu",
        weights_only=True
    )

    return (
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"]
    )

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()
    print("X_train", tuple(X_train.shape), "| X_val", tuple(X_val.shape), "| X_test", tuple(X_test.shape))
    print("So mau moi lop (train):", torch.bincount(y_train).tolist())
    # print("Working directory:", Path.cwd())
    # print("Processed folder:", PROCESSED_PATH.resolve())

    # file_path = PROCESSED_PATH / "mnist.pt"
    # print("File path:", file_path.resolve())
    # print("File exists:", file_path.is_file())

    # if PROCESSED_PATH.exists():
    #     print("Folder contents:", list(PROCESSED_PATH.iterdir()))