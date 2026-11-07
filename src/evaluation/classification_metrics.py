"""Classification metrics computed from the confusion matrix (NumPy only).

For the confusion matrix ``C`` (rows = true class, columns = predicted class):
    TP_k = C[k, k],  FP_k = sum_i C[i, k] - TP_k,  FN_k = sum_j C[k, j] - TP_k
    precision_k = TP_k / (TP_k + FP_k),  recall_k = TP_k / (TP_k + FN_k),  F1_k = 2PR / (P + R)
Macro metrics are the unweighted mean over the classes; a zero denominator gives 0.
scikit-learn is used only in ``tests/test_metrics.py`` as an independent oracle.
"""

import numpy as np


def _to_numpy(y):
    """Convert labels (list / ndarray / torch tensor) into a 1D NumPy array."""
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    elif y.ndim != 1:
        raise ValueError("y_true và y_pred phải chứa nhãn lớp dạng 1D.")
    return y


def confusion_matrix(y_true, y_pred, labels):
    """``C[i, j]`` = number of samples with true label ``labels[i]`` predicted as ``labels[j]``.

    Samples whose label (true or predicted) is not in ``labels`` are ignored.
    """
    labels = np.asarray(list(labels))
    pos = {int(lab): i for i, lab in enumerate(labels)}
    k = len(labels)
    yt = np.array([pos.get(int(v), -1) for v in y_true])
    yp = np.array([pos.get(int(v), -1) for v in y_pred])
    ok = (yt >= 0) & (yp >= 0)
    return np.bincount(yt[ok] * k + yp[ok], minlength=k * k).reshape(k, k)


def per_class_metrics(cm):
    """Per-class precision, recall, F1 and support from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    pred_pos, support = cm.sum(axis=0), cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_pos > 0, tp / pred_pos, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return precision, recall, f1, support.astype(int)


def format_report(cm, labels):
    precision, recall, f1, support = per_class_metrics(cm)
    lines = [f"{'':>12}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}", ""]
    for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True):
        lines.append(f"{label!s:>12}{p:>10.2f}{r:>10.2f}{f:>10.2f}{s:>10d}")
    total = int(support.sum())
    acc = float(np.trace(cm) / total) if total else 0.0
    lines += [
        "",
        f"{'accuracy':>12}{'':>20}{acc:>10.2f}{total:>10d}",
        f"{'macro avg':>12}{precision.mean():>10.2f}{recall.mean():>10.2f}"
        f"{f1.mean():>10.2f}{total:>10d}",
    ]
    return "\n".join(lines)


def get_metrics(y_true, y_pred, labels=None):
    """Accuracy, macro precision / recall / F1, confusion matrix and a text report."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true và y_pred phải có cùng số mẫu.")
    if len(y_true) == 0:
        raise ValueError("y_true và y_pred không được rỗng.")
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else list(labels)

    cm = confusion_matrix(y_true, y_pred, labels)
    precision, recall, f1, _ = per_class_metrics(cm)
    return {
        "accuracy": float(np.trace(cm) / len(y_true)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "confusion_matrix": cm,
        "report": format_report(cm, labels),
    }
