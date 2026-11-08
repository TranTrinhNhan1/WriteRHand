"""Stacking ensemble: a softmax-regression meta-learner on top of already trained base classifiers.

Level 0   base models (Gaussian NB, softmax, MLP, AdaBoost, CNN, linear SVM), each trained on the train split.
Features  for every base model and image: log of the clipped class probabilities (``predict_proba``), or the raw
          one-vs-rest scores for the SVM (``decision_function``). The concatenation has K * n_models columns
          and is standardised with the mean / std of the meta-training features.
Level 1   ``SoftmaxRegression`` (own implementation) trained on those features.

The meta-learner is fitted on a *hold-out* split (the validation split), never on the images the base models were
trained on: their outputs on their own training data are over-confident (AdaBoost reaches 100 %), so a meta-learner
fitted there would simply trust them. The test split is used for reporting only.
"""

import numpy as np

from src.models.base import BaseClassifier, check_X
from src.models.logistic_regression import SoftmaxRegression

PROBA_EPS = 1e-4


def base_scores(model, X, eps=PROBA_EPS):
    """Level-0 output of one base model: log-probabilities if available, else decision scores."""
    if hasattr(model, "predict_proba"):
        try:
            return np.log(np.clip(model.predict_proba(X), eps, 1.0))
        except NotImplementedError:
            pass
    return model.decision_function(X)


class StackingEnsemble(BaseClassifier):
    """Meta-learner over fitted base models (``bases`` is a list of ``(name, fitted_model)``)."""

    _state_arrays = ("classes", "W", "b", "mu", "sd")

    def __init__(self, bases=None, l2=1e-3, learning_rate=0.1, epochs=60, batch_size=128, seed=42):
        super().__init__()
        self.bases = list(bases or [])
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def get_params(self):
        return {
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "bases": [name for name, _ in self.bases],
        }

    # ------------------------------------------------------------------ features
    def features(self, X):
        """Raw (unstandardised) level-1 features of ``X``: one block of K columns per base model."""
        if not self.bases:
            raise ValueError("StackingEnsemble has no base models.")
        X = check_X(X)
        return np.hstack([base_scores(m, X) for _, m in self.bases])

    def block_slices(self):
        k = len(self.classes)
        return {name: slice(i * k, (i + 1) * k) for i, (name, _) in enumerate(self.bases)}

    # ------------------------------------------------------------------ training
    def fit_features(self, F, y):
        """Fit the meta-learner on precomputed level-1 features ``F`` (N, K * n_models)."""
        F = check_X(F, name="F")
        self.mu = F.mean(axis=0)
        self.sd = F.std(axis=0) + 1e-8
        meta = SoftmaxRegression(
            learning_rate=self.learning_rate,
            l2=self.l2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            seed=self.seed,
        ).fit((F - self.mu) / self.sd, y)
        self.classes, self.W, self.b = meta.classes, meta.W, meta.b
        self.history = meta.history
        return self

    def fit(self, X, y):
        """``X``, ``y`` must be a hold-out split the base models did not train on."""
        return self.fit_features(self.features(X), y)

    # ------------------------------------------------------------------ inference
    def logits_from_features(self, F):
        self._check_fitted()
        return ((F - self.mu) / self.sd) @ self.W + self.b

    def predict_from_features(self, F):
        return self.classes[self.logits_from_features(F).argmax(axis=1)]

    def predict_proba(self, X):
        z = self.logits_from_features(self.features(X))
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_from_features(self.features(X))

    def block_importance(self):
        """Mean |weight| of each base model's block in the meta-learner (standardised inputs)."""
        return {n: float(np.abs(self.W[s]).mean()) for n, s in self.block_slices().items()}

    # ------------------------------------------------------------------ persistence (meta-learner only)
    def _extra_state(self):
        return {"bases": [n for n, _ in self.bases]}

    @classmethod
    def load_with_bases(cls, path, bases):
        """Restore a saved meta-learner; ``bases`` must be the same models, in the same order."""
        import json

        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["__meta__"]))
            names = meta["extra"]["bases"]
            if names != [n for n, _ in bases]:
                raise ValueError(
                    f"base models {[n for n, _ in bases]} do not match the saved {names}"
                )
            p = {k: v for k, v in meta["params"].items() if k != "bases"}
            model = cls(bases=bases, **p)
            model._set_arrays({k[4:]: f[k] for k in f.files if k.startswith("arr_")})
        return model
