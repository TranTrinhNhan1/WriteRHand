"""Multiclass linear SVM (one-vs-rest) trained with mini-batch gradient descent.

Model        z = X W + b            (one linear score per class), prediction = argmax_k z_k
Targets      t_ik = +1 if y_i = k else -1
Loss         L(W, b) = 1/B sum_i sum_k phi(1 - t_ik z_ik)  +  (l2 / 2) * ||W||_F^2      (bias not penalised)
             hinge          phi(m) = max(0, m)            (sub-gradient)
             squared hinge  phi(m) = max(0, m)^2          (differentiable, the "L2-SVM")
Gradient     dL/dZ = -t * phi'(1 - t z) / B,   dL/dW = X^T dL/dZ + l2 * W,   dL/db = sum_i dL/dZ_i

Everything is plain NumPy. scikit-learn (``LinearSVC``) is used only in the tests / comparison scripts.
"""

import numpy as np

from src.models.base import BaseClassifier, check_X, check_Xy

LOSSES = ("hinge", "squared_hinge")


def hinge_terms(Z, T, loss):
    """Per-element loss ``phi(1 - t z)`` and its derivative with respect to ``z`` (before the 1/B)."""
    margin = 1.0 - T * Z
    active = margin > 0
    if loss == "hinge":
        return np.where(active, margin, 0.0), np.where(active, -T, 0.0)
    return np.where(active, margin * margin, 0.0), np.where(active, -2.0 * T * margin, 0.0)


class LinearSVM(BaseClassifier):
    """One-vs-rest linear SVM with L2 regularisation, trained by mini-batch gradient descent.

    Parameters
    ----------
    learning_rate : step size (decays as ``lr / (1 + lr_decay * epoch)``).
    l2 : coefficient of the (l2/2)*||W||^2 penalty (plays the role of 1/(C n) in the usual SVM notation).
    loss : ``"hinge"`` or ``"squared_hinge"``.
    epochs, batch_size, tol, n_iter_no_change, seed : as in ``SoftmaxRegression``.
    """

    _state_arrays = ("classes", "W", "b")

    def __init__(
        self,
        learning_rate=0.01,
        l2=1e-4,
        loss="hinge",
        epochs=30,
        batch_size=128,
        lr_decay=0.0,
        tol=None,
        n_iter_no_change=3,
        seed=42,
        verbose=False,
    ):
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if l2 < 0:
            raise ValueError("l2 must be >= 0.")
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}.")
        if epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be >= 1 or None.")
        self.learning_rate = learning_rate
        self.l2 = l2
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------ maths
    @staticmethod
    def loss_and_grad(W, b, X, Y, l2=0.0, loss="hinge"):
        """Loss and analytic (sub-)gradients on a batch. ``Y`` is one-hot (B, K)."""
        B = X.shape[0]
        Z = X @ W + b
        T = 2.0 * Y - 1.0
        phi, dphi = hinge_terms(Z, T, loss)
        value = float(phi.sum() / B) + 0.5 * l2 * float((W * W).sum())
        dZ = dphi / B
        return value, X.T @ dZ + l2 * W, dZ.sum(axis=0)

    def decision_function(self, X):
        """Raw one-vs-rest scores ``X W + b``, shape (N, K)."""
        self._check_fitted()
        X = check_X(X, self.W.shape[0])
        return X @ self.W + self.b

    # ------------------------------------------------------------------ training
    def fit(self, X, y, X_val=None, y_val=None, callback=None):
        """Train. ``X_val``/``y_val`` are only *monitored* (history), never used for decisions."""
        X, y = check_Xy(X, y)
        self.classes, y_idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        k = len(self.classes)
        Y = np.eye(k)[y_idx]
        rng = np.random.default_rng(self.seed)
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        bs = n if self.batch_size is None else min(self.batch_size, n)

        have_val = X_val is not None and y_val is not None
        if have_val:
            X_val = check_X(X_val, d, "X_val")
            yv_idx = np.searchsorted(self.classes, np.asarray(y_val))
            Yv = np.eye(k)[yv_idx]
        self.history = {"loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        best, wait = np.inf, 0
        if callback is not None:
            callback(0, self)

        for epoch in range(self.epochs):
            lr = self.learning_rate / (1.0 + self.lr_decay * epoch)
            order = rng.permutation(n)
            for s in range(0, n, bs):
                idx = order[s : s + bs]
                _, gW, gb = self.loss_and_grad(self.W, self.b, X[idx], Y[idx], self.l2, self.loss)
                self.W -= lr * gW
                self.b -= lr * gb
            Z = X @ self.W + self.b
            phi, _ = hinge_terms(Z, 2.0 * Y - 1.0, self.loss)
            value = float(phi.sum() / n) + 0.5 * self.l2 * float((self.W**2).sum())
            if not np.isfinite(value):
                raise FloatingPointError("Loss became NaN/inf - lower the learning rate.")
            self.history["loss"].append(value)
            self.history["train_acc"].append(float(np.mean(Z.argmax(1) == y_idx)))
            self.history["lr"].append(lr)
            msg = f"epoch {epoch + 1}/{self.epochs} loss={value:.4f}"
            if have_val:
                Zv = X_val @ self.W + self.b
                phi_v, _ = hinge_terms(Zv, 2.0 * Yv - 1.0, self.loss)
                self.history["val_loss"].append(float(phi_v.sum() / len(yv_idx)))
                self.history["val_acc"].append(float(np.mean(Zv.argmax(1) == yv_idx)))
                msg += f" val_acc={self.history['val_acc'][-1]:.4f}"
            if self.verbose:
                print(msg)
            if callback is not None:
                callback(epoch + 1, self)
            if self.tol is not None:
                if best - value > self.tol:
                    best, wait = value, 0
                else:
                    wait += 1
                    if wait >= self.n_iter_no_change:
                        break
        self.n_iter_ = len(self.history["loss"])
        return self

    # ------------------------------------------------------------------ inference
    def predict(self, X):
        scores = self.decision_function(X)  # raises NotFittedError before ``classes`` is touched
        return self.classes[scores.argmax(axis=1)]

    def _extra_state(self):
        return {"n_iter": int(getattr(self, "n_iter_", 0))}

    def _set_extra_state(self, extra):
        self.n_iter_ = extra.get("n_iter", 0)
