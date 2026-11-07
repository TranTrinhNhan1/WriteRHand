import json
import time

import numpy as np
import torch

from src.evaluation.classification_metrics import get_metrics
from src.utils.paths import RESULTS_DIR


def _json_default(obj):
    """Chuyen kieu np thanh kieu co the luu json"""
    if isinstance(obj, np.ndarray):  # neu la mang nhieu chieu
        return obj.tolist()
    if isinstance(obj, np.generic):  # neu la kieu co so nhu np.int32
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Không thể lưu kiểu {type(obj)} vào JSON")


class Evaluator:
    def __init__(self, name, model, num_classes=10):
        self.name = name
        self.model = model
        self.num_classes = num_classes

        self.results = {
            "model_name": name,
            "num_classes": num_classes,
            "train_time_sec": None,
            "splits": {},
        }

    def fit(self, X_tr, y_tr, **fit_kwargs):
        """Train and record wall-clock time. ``fit_kwargs`` e.g. X_val / y_val (monitoring only)."""
        start = time.perf_counter()
        self.model.fit(X_tr, y_tr, **fit_kwargs)

        elapsed = time.perf_counter() - start
        self.results["train_time_sec"] = elapsed

        return self

    def evaluate(self, X, y, split_name, repeats=1):
        """Metrics on one split. Inference time = median wall-clock of ``repeats`` full passes."""
        if split_name not in {"train", "val", "test"}:
            raise ValueError("split_name phai la 'train', 'val' hoac 'test'")

        times = []
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            y_pred = self.model.predict(X)
            times.append(time.perf_counter() - start)
        predict_time = float(np.median(times))
        self.predictions = getattr(self, "predictions", {})
        self.predictions[split_name] = np.asarray(y_pred)

        metrics = get_metrics(y, y_pred, labels=range(self.num_classes))

        metrics["predict_time_sec"] = predict_time
        metrics["predict_time_ms_per_sample"] = 1000.0 * predict_time / len(y_pred)
        metrics["n_samples"] = int(len(y_pred))

        self.results["splits"][split_name] = metrics

        return metrics

    def eval_all(self, X_tr, y_tr, X_val, y_val, X_te, y_te):
        self.evaluate(X_tr, y_tr, "train")
        self.evaluate(X_val, y_val, "val")
        if X_te is not None and y_te is not None:
            self.evaluate(X_te, y_te, "test")

        return self.results

    def save(self, folder=RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.name}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.results, file, ensure_ascii=False, indent=2, default=_json_default)
