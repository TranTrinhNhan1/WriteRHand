import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.evaluation.classification_metrics import get_metrics


def test_hand_computed_example():
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 1, 0]
    m = get_metrics(y_true, y_pred, labels=range(3))
    np.testing.assert_array_equal(m["confusion_matrix"], [[2, 1, 0], [0, 2, 0], [1, 0, 0]])
    assert m["accuracy"] == pytest.approx(4 / 6)
    # precision: [2/3, 2/3, 0], recall: [2/3, 1, 0]
    assert m["precision_macro"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert m["recall_macro"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    f1 = [2 * (2 / 3) * (2 / 3) / (4 / 3), 2 * (2 / 3) * 1 / (5 / 3), 0.0]
    assert m["f1_macro"] == pytest.approx(np.mean(f1))


def test_matches_scikit_learn_on_random_predictions():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 10, 2000)
    y_pred = np.where(rng.random(2000) < 0.7, y_true, rng.integers(0, 10, 2000))
    m = get_metrics(y_true, y_pred, labels=range(10))
    kw = {"labels": range(10), "average": "macro", "zero_division": 0}
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["precision_macro"] == pytest.approx(precision_score(y_true, y_pred, **kw))
    assert m["recall_macro"] == pytest.approx(recall_score(y_true, y_pred, **kw))
    assert m["f1_macro"] == pytest.approx(f1_score(y_true, y_pred, **kw))
    np.testing.assert_array_equal(
        m["confusion_matrix"], confusion_matrix(y_true, y_pred, labels=range(10))
    )


def test_class_never_predicted_gives_zero_not_nan():
    m = get_metrics([0, 1, 2], [0, 1, 1], labels=range(3))
    assert np.isfinite(m["precision_macro"]) and np.isfinite(m["f1_macro"])
    assert m["precision_macro"] == pytest.approx((1 + 0.5 + 0) / 3)


def test_accepts_torch_tensors_and_checks_shapes():
    import torch

    m = get_metrics(torch.tensor([0, 1]), torch.tensor([0, 1]), labels=range(2))
    assert m["accuracy"] == 1.0
    with pytest.raises(ValueError):
        get_metrics([0, 1], [0], labels=range(2))
    with pytest.raises(ValueError):
        get_metrics([], [], labels=range(2))
