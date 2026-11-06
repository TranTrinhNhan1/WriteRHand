import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def _to_numpy(y):
    """Chuyển nhãn từ PyTorch Tensor hoặc dữ liệu khác thành NumPy 1D."""
    if hasattr(y, "detach"):  # tensor pt co method detach, np thi khong
        y = y.detach().cpu().numpy() # detach tach y ra khoi computational graph (ham tinh dao ham)

    y = np.asarray(y)

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel() # bien (n, 1) thanh 1D array neu co
    elif y.ndim != 1:
        raise ValueError("y_true và y_pred phải chứa nhãn lớp dạng 1D.")

    return y


def get_metrics(y_true, y_pred, labels=None): # labels la range(10)
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true và y_pred phải có cùng số mẫu.")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(
            y_true, y_pred, labels=labels,
            average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=labels,
            average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, labels=labels,
            average="macro", zero_division=0
        ),

        # "precision_weighted": precision_score(
        #     y_true, y_pred, labels=labels,
        #     average="weighted", zero_division=0
        # ),
        # "recall_weighted": recall_score(
        #     y_true, y_pred, labels=labels,
        #     average="weighted", zero_division=0
        # ),
        # "f1_weighted": f1_score(
        #     y_true, y_pred, labels=labels,
        #     average="weighted", zero_division=0
        # ),

        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ),
        "report": classification_report(
            y_true, y_pred, labels=labels,
            zero_division=0
        ),
    }

    return metrics
