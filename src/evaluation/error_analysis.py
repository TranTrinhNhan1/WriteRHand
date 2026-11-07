"""
Cần đảm bảo các model khác nhau dự đoán trên cùng một thứ tự mẫu mới có thể áp dụng file .py này
True means incorrect, False means correct
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_error_mask(y_true, y_pred):
    """
    Return boolean mask indicating the wrong predictions
    True means incorrect, False means correct
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred has different dim values, they must be 1D arrays")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred has different lengths")

    return y_true != y_pred


def make_error_matrix(y_true, predictions):
    """
    predictions: a dict of {model_name: y_pred}
    returns Boolean error (N, M) matrix
    each column is a model
    """
    if not predictions:
        raise ValueError("predictions is empty!")
    model_names = list(predictions.keys())
    error_masks = []

    for model in model_names:
        y_pred = predictions[model]
        error_masks.append(get_error_mask(y_true, y_pred))
    E = np.column_stack(error_masks)  # ghep nhieu 1D array thanh cac cot cua ma tran

    return E, model_names


def summarize_error(error_matrix, model_names):
    """
    return dict containing aggregate error statistics
    at_least_one_correct is the crucial metric to measure diversity for ensemble, the higher the more suitable
    """
    error_matrix = np.asarray(error_matrix, dtype=bool)

    if error_matrix.ndim != 2:
        raise ValueError("error_matrix must be a 2D matrix")
    n_samples, n_models = error_matrix.shape

    if len(model_names) != n_models:
        raise ValueError("Number of model names must match error_matrix columns")

    errors_per_model = np.sum(error_matrix, axis=0)
    all_wrong_mask = np.all(error_matrix, axis=1)
    all_correct_mask = ~np.any(error_matrix, axis=1)
    at_least_one_correct = ~all_wrong_mask
    errors_per_sample = np.sum(error_matrix, axis=1)

    return {
        "error_counts": {
            name: int(count) for name, count in zip(model_names, errors_per_model, strict=True)
        },
        "all_wrong_count": int(np.sum(all_wrong_mask)),
        "all_correct_count": int(np.sum(all_correct_mask)),
        "oracle_accuracy": float(np.mean(at_least_one_correct)),
        "errors_per_sample": errors_per_sample,
    }


def compare_pair_errors(mask_a, mask_b):
    """
    return the four counts of correct/error - a/b matrix
    """
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    if mask_a.ndim != 1 or mask_b.ndim != 1:
        raise ValueError("Both error masks must be 1D arrays.")

    if len(mask_a) != len(mask_b):
        raise ValueError("Both error masks must have the same length.")

    ab_correct = np.sum(~mask_a & ~mask_b)
    a_error_b_correct = np.sum(mask_a & ~mask_b)
    a_correct_b_error = np.sum(~mask_a & mask_b)
    ab_error = np.sum(mask_a & mask_b)

    return int(ab_correct), int(a_error_b_correct), int(a_correct_b_error), int(ab_error)


if __name__ == "__main__":
    # 1. Load and split Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 2. Define models
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    }

    # 3. Train models and collect predictions
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

    # 4. Create error matrix
    E, model_names = make_error_matrix(y_test, predictions)

    # print("\nError matrix:")
    # print(E.astype(int))
    # print("Model names:", model_names)

    # 5. Aggregate error statistics
    stats = summarize_error(E, model_names)

    print("\nError statistics:")
    print("Errors per model:", stats["error_counts"])
    print("All models wrong:", stats["all_wrong_count"])
    print("All models correct:", stats["all_correct_count"])
    print("Oracle accuracy:", stats["oracle_accuracy"])

    print("\nErrors per sample:")
    print(stats["errors_per_sample"])

    # 6. Compare the two models
    mask_a = E[:, 0]
    mask_b = E[:, 1]

    a, b, c, d = compare_pair_errors(mask_a, mask_b)

    n_samples = len(y_test)
    disagreement_rate = (b + c) / n_samples

    print("\nPair comparison:")
    print(f"Model A: {model_names[0]}")
    print(f"Model B: {model_names[1]}")
    print(f"Both correct (a): {a}")
    print(f"A wrong, B correct (b): {b}")
    print(f"A correct, B wrong (c): {c}")
    print(f"Both wrong (d): {d}")
    print(f"Disagreement count: {b + c}")
    print(f"Disagreement rate: {disagreement_rate:.4f}")
