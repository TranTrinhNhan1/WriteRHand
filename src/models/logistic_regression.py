"""Multiclass logistic (softmax) regression trained with mini-batch gradient descent.

Model        z = X W + b,   p = softmax(z)
Loss         L(W, b) = -1/B sum_i log p_i[y_i]  +  (l2 / 2) * ||W||_F^2      (bias not penalised)
Gradient     dL/dZ = (P - Y) / B,    dL/dW = X^T dL/dZ + l2 * W,    dL/db = sum_i dL/dZ_i

Everything is plain NumPy. scikit-learn is used only in the tests / comparison scripts.
"""

import numpy as np

from src.models.base import BaseClassifier, check_X, check_Xy


def softmax(z):
    """Row-wise softmax, stable for large |z| (subtracts the row max)."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def cross_entropy(logits, y_idx):
    """Mean negative log-likelihood from raw logits (log-sum-exp form, never takes log(0))."""
    return float(-log_softmax(logits)[np.arange(len(y_idx)), y_idx].mean())


class SoftmaxRegression(BaseClassifier):
    """Softmax regression with L2 regularisation and mini-batch gradient descent.

    Parameters
    ----------
    learning_rate : step size of gradient descent (decays as ``lr / (1 + lr_decay * epoch)``).
    l2 : coefficient of the (l2/2)*||W||^2 penalty.
    epochs : maximum number of passes over the training data.
    batch_size : mini-batch size (``None`` -> full batch).
    tol, n_iter_no_change : stop when the epoch loss improved by less than ``tol`` for
        ``n_iter_no_change`` consecutive epochs (``tol=None`` disables the check).
    seed : seed of the shuffling RNG.
    """

    _state_arrays = ("classes", "W", "b")

    def __init__(
        self,
        learning_rate=0.1,
        l2=0.0,
        epochs=50,
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
        if epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be >= 1 or None.")
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------ maths
    @staticmethod
    def loss_and_grad(W, b, X, Y, l2=0.0):
        """Loss and analytic gradients on a batch. ``Y`` is one-hot (B, K)."""
        B = X.shape[0]
        Z = X @ W + b
        loss = cross_entropy(Z, Y.argmax(axis=1)) + 0.5 * l2 * float((W * W).sum())
        dZ = (softmax(Z) - Y) / B
        return loss, X.T @ dZ + l2 * W, dZ.sum(axis=0)

    def _logits(self, X):
        self._check_fitted()
        X = check_X(X, self.W.shape[0])
        return X @ self.W + self.b

    # ------------------------------------------------------------------ training
    def fit(self, X, y, X_val=None, y_val=None, callback=None):
        """Train. ``X_val``/``y_val`` are only *monitored* (history), never used for decisions.

        ``callback(epoch, model)`` (optional, used to record training replays) is called with epoch 0
        after initialisation and after every epoch; it must not modify the model."""
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
        self.history = {"loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        best, wait = np.inf, 0
        if callback is not None:
            callback(0, self)

        for epoch in range(self.epochs):
            lr = self.learning_rate / (1.0 + self.lr_decay * epoch)
            order = rng.permutation(n)
            for s in range(0, n, bs):
                idx = order[s : s + bs]
                _, gW, gb = self.loss_and_grad(self.W, self.b, X[idx], Y[idx], self.l2)
                self.W -= lr * gW
                self.b -= lr * gb
            Z = X @ self.W + self.b
            loss = cross_entropy(Z, y_idx) + 0.5 * self.l2 * float((self.W**2).sum())
            if not np.isfinite(loss):
                raise FloatingPointError("Loss became NaN/inf - lower the learning rate.")
            self.history["loss"].append(loss)
            self.history["train_acc"].append(float(np.mean(Z.argmax(1) == y_idx)))
            self.history["lr"].append(lr)
            msg = f"epoch {epoch + 1}/{self.epochs} loss={loss:.4f}"
            if have_val:
                Zv = X_val @ self.W + self.b
                self.history["val_loss"].append(cross_entropy(Zv, yv_idx))
                self.history["val_acc"].append(float(np.mean(Zv.argmax(1) == yv_idx)))
                msg += f" val_acc={self.history['val_acc'][-1]:.4f}"
            if self.verbose:
                print(msg)
            if callback is not None:
                callback(epoch + 1, self)
            if self.tol is not None:
                if best - loss > self.tol:
                    best, wait = loss, 0
                else:
                    wait += 1
                    if wait >= self.n_iter_no_change:
                        break
        self.n_iter_ = len(self.history["loss"])
        return self

    # ------------------------------------------------------------------ inference
    def predict_proba(self, X):
        return softmax(self._logits(X))

    def predict_log_proba(self, X):
        return log_softmax(self._logits(X))

    def predict(self, X):
        idx = self._logits(X).argmax(axis=1)
        return self.classes[idx]

    def _extra_state(self):
        return {"n_iter": int(getattr(self, "n_iter_", 0))}

    def _set_extra_state(self, extra):
        self.n_iter_ = extra.get("n_iter", 0)
