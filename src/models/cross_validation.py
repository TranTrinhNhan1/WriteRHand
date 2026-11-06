'''
Dùng tập train để train K-fold tìm bộ siêu tham số tốt nhất
Cần preprocess lại từ đầu vì tập train đã chuẩn hóa theo mean/std, nếu dùng thì sẽ gây data leakage trên các fold.
'''
from sklearn.linear_model import LogisticRegression

from evaluation.classification_metrics import get_metrics
from preprocessing.preprocess import prepare_mnist, load_mnist
import numpy as np
import torch
from itertools import product


SEED = 42


def create_folds(n_samples, k=4, seed=SEED):
    ''' tra ve cac chi so cua tap train va cua tap val'''
    if k < 2 or k > n_samples:
        raise ValueError("k phải nằm trong khoảng [2, n_samples].")

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(n_samples, generator=generator)

    folds = torch.tensor_split(shuffled_indices, k) # chia thanh k phan bang nhau
    splits = []

    for i in range(k):
        # moi lan lap lay 1 fold lam val va cac fold con lai lam train
        val_indices = folds[i]

        train_indices = torch.cat([
            folds[j] for j in range(k) if j != i
        ])

        splits.append((train_indices, val_indices))

    return splits

def run_kfold(model_factory, X, y, K=4, seed=SEED, labels=None): 
    '''    labels la range(10)
    model_factory la ham tao model moi
    X, y la data chua preprocess
    '''
    splits = create_folds(len(X), K, seed)
    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro"
    ]

    fold_scores = []
    for fold_no, (idx_train, idx_val) in enumerate(splits, start=1):
        # ban dau la raw, lay idx de chia train/val
        X_tr = X[idx_train]
        y_tr = y[idx_train]
        X_val = X[idx_val]
        y_val = y[idx_val]

        # preprocess de thanh du lieu chuan
        X_tr, y_tr, X_val, y_val, _, _ = prepare_mnist(X_tr, y_tr, X_val, y_val, has_validation=False, n_train=len(X))
        model = model_factory()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        metrics = get_metrics(y_val, y_pred, labels=labels)
        scores = {name: metrics[name] for name in metric_names}

        fold_scores.append(scores)
        print(f"Fold {fold_no}: acc={scores["accuracy"]:.3f}")

        mean_scores = {}
        std_scores = {}

        for name in metric_names:
            values = torch.tensor(
                [scores[name] for scores in fold_scores],
                dtype=torch.float64
            )
            mean_scores[name] = values.mean().item()
            std_scores[name] = values.std(unbiased=False).item()

    return mean_scores, std_scores, fold_scores

def find_grid(
    model_class, param_grid, X, y,
    K=4, seed=SEED, scoring="accuracy", labels=None
):
    '''
    tra ve du lieu de chon cau hinh sieu tham so
    co the ve bieu do
    '''
    if not param_grid or \
        any(len(values) == 0 for values in param_grid.values()):
        raise ValueError("Luoi sieu tham so bi rong!")
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    results = []
    best_params = None
    best_score = float("-inf")

    for values in product(*param_values): # duyet tat ca to hop param
        cur_params = dict(zip(param_names, values))
        # ham tao model moi cho moi bo param
        model_factory = lambda : model_class(**cur_params)

        mean_scores, std_scores, fold_scores = run_kfold(
            model_factory, X, y, K, seed, labels
        )
        cur_res = {
            "params": cur_params.copy(),
            "mean_scores": mean_scores,
            "std_scores": std_scores,
            "fold_scores": fold_scores
        }
        results.append(cur_res)

        # tim diem cao nhat
        score = mean_scores[scoring] # lay metric chinh de so
        if score > best_score:
            best_score = score
            best_params = cur_params.copy()

    return best_params, results


if __name__ == "__main__":
    Xtr_raw, ytr_raw, _, _ = load_mnist()

    X_train, y_train = Xtr_raw[:10000], ytr_raw[:10000]

    # 3. Khai báo lưới siêu tham số
    param_grid = {
        "C": [0.01, 0.1, 1.0],
        "max_iter": [300]
    }

    # 4. Tìm cấu hình tốt nhất bằng K-fold
    best_params, results = find_grid(
        model_class=LogisticRegression,
        param_grid=param_grid,
        X=X_train,
        y=y_train,
        K=5,
        seed=SEED,
        scoring="accuracy",
        labels=range(10)
    )

    # 5. In kết quả
    print("\nCấu hình tốt nhất:")
    print(best_params)

    print("\nKết quả từng cấu hình:")
    for result in results:
        print(
            f"Params: {result['params']}, "
            f"Accuracy: {result['mean_scores']['accuracy']:.4f}, "
            f"Std: {result['std_scores']['accuracy']:.4f}"
        )