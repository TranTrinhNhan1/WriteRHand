"""Common interface of every classifier in the project.

Conventions (see WORKFLOW.md): ``X`` has shape ``(N, 784)``, ``y`` holds integer labels,
``fit`` returns ``self``, hyper-parameters are passed to ``__init__``, ``predict`` returns labels.

Persistence: ``save(path)`` writes ONE ``.npz`` file (no pickle) that holds the constructor
parameters, the fitted arrays and the model class name; ``BaseClassifier.load(path)`` restores
the right class.
"""

import inspect
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class NotFittedError(RuntimeError):
    pass


def check_X(X, n_features=None, name="X"):
    """Validate a feature matrix and return it as a float64 ndarray."""
    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must contain numeric values.") from e
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (N, D).")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} must not contain NaN or infinity.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"{name} has {X.shape[1]} features but the model was fitted with {n_features}.")
    return X


def check_Xy(X, y):
    X = check_X(X)
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")
    return X, y


class BaseClassifier(ABC):
    #: names of fitted arrays stored by ``save`` (subclasses list their attributes here)
    _state_arrays: tuple = ()

    def __init__(self):
        self.history = {}

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def predict_proba(self, X):
        raise NotImplementedError(f"{self.__class__.__name__} does not support predict_proba().")

    # ------------------------------------------------------------------ parameters
    @classmethod
    def _param_names(cls):
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters if p != "self"]

    def get_params(self):
        """Constructor hyper-parameters (JSON friendly)."""
        out = {}
        for name in self._param_names():
            v = getattr(self, name)
            out[name] = list(v) if isinstance(v, tuple) else v
        return out

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self._param_names():
                raise ValueError(f"Unknown parameter {k!r} for {self.__class__.__name__}.")
            setattr(self, k, tuple(v) if isinstance(getattr(self, k), tuple) else v)
        return self

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ------------------------------------------------------------------ persistence
    def _check_fitted(self):
        if not hasattr(self, "classes"):
            raise NotFittedError(f"{self.__class__.__name__} is not fitted yet.")

    def _get_arrays(self):
        """Fitted arrays to persist, as ``{name: ndarray}`` (override for ragged state)."""
        return {n: np.asarray(getattr(self, n)) for n in self._state_arrays}

    def _set_arrays(self, arrays):
        for n, v in arrays.items():
            setattr(self, n, v)

    def _extra_state(self):
        """Extra JSON-serialisable fitted values (override when needed)."""
        return {}

    def _set_extra_state(self, extra):
        pass

    def save(self, path):
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "class": self.__class__.__name__,
            "params": self.get_params(),
            "extra": self._extra_state(),
        }
        arrays = {f"arr_{n}": v for n, v in self._get_arrays().items()}
        np.savez_compressed(path, __meta__=np.array(json.dumps(meta)), **arrays)
        return path

    @staticmethod
    def load(path):
        # imported lazily to avoid circular imports
        from src.models import registry

        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["__meta__"]))
            cls = registry.MODELS[meta["class"]]
            model = cls(**meta["params"])
            model._set_arrays({k[4:]: f[k] for k in f.files if k.startswith("arr_")})
        model._set_extra_state(meta["extra"])
        return model
