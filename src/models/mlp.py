"""Multilayer perceptron trained with hand-written backpropagation.

Architecture   x -> [Linear -> ReLU (-> inverted dropout)] * H -> Linear -> softmax
Loss           mean cross-entropy
Backprop       delta_L = (softmax(z_L) - onehot(y)) / B
               dW_l = a_{l-1}^T delta_l ,   db_l = sum_rows(delta_l)
               delta_{l-1} = (delta_l W_l^T) * 1[a_{l-1} > 0] * dropout_mask_{l-1}
Update         SGD with momentum:  v <- mu * v + g ;  theta <- theta - lr * v   (explicit code)
Initialisation He normal (std = sqrt(2 / fan_in)), zero biases, drawn from a private seeded RNG.

PyTorch is used only as a tensor library (matmul, random numbers): *autograd is never used for
training* and no ``nn.Module`` / ``torch.optim`` object is involved. ``tests/test_mlp.py`` checks
the manual gradients against finite differences and against autograd.

Early stopping uses an *internal* hold-out (``val_fraction`` of the training data), so the
project's validation split never influences training.
"""

import numpy as np
import torch

from src.models.base import BaseClassifier, NotFittedError, check_X, check_Xy

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


class MLP(BaseClassifier):
    def __init__(
        self,
        hidden_sizes=(128,),
        *,
        learning_rate=0.1,
        p_drop=0.0,
        epochs=100,
        batch_size=128,
        momentum=0.0,
        seed=42,
        early_stopping=True,
        val_fraction=0.08,
        patience=5,
        min_delta=1e-4,
        dtype="float32",
        verbose=False,
    ):
        super().__init__()
        hidden_sizes = tuple(int(h) for h in hidden_sizes)
        if any(h < 1 for h in hidden_sizes):
            raise ValueError("hidden_sizes must be positive integers.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if not 0.0 <= p_drop < 1.0:
            raise ValueError("p_drop must be in [0, 1).")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1.")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {list(_DTYPES)}.")
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.p_drop = p_drop
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.seed = seed
        self.early_stopping = early_stopping
        self.val_fraction = val_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.dtype = dtype
        self.verbose = verbose

    # ------------------------------------------------------------------ parameters
    @property
    def _tdtype(self):
        return _DTYPES[self.dtype]

    def _init_params(self, n_features, n_classes):
        """He-normal weights ``W_l`` of shape (fan_in, fan_out) and zero biases."""
        sizes = [n_features, *self.hidden_sizes, n_classes]
        gen = torch.Generator().manual_seed(self.seed)
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
            std = (2.0 / fan_in) ** 0.5
            self.W.append(
                (torch.randn(fan_in, fan_out, generator=gen, dtype=torch.float64) * std).to(
                    self._tdtype
                )
            )
            self.b.append(torch.zeros(fan_out, dtype=self._tdtype))
        self.generator = gen  # continues the same stream for shuffling and dropout

    def _check_fitted(self):
        if not hasattr(self, "W"):
            raise NotFittedError("MLP is not fitted yet.")

    # ------------------------------------------------------------------ forward / backward
    def _forward_cache(self, xb, training=False):
        """Return (inputs_of_each_layer, dropout_masks, logits)."""
        inputs, masks = [], []
        a = xb
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            inputs.append(a)
            z = a @ W + b
            mask = None
            if i < last:
                a = torch.relu(z)
                if training and self.p_drop > 0:
                    keep = (
                        torch.rand(a.shape, generator=self.generator, dtype=a.dtype) >= self.p_drop
                    ).to(a.dtype)
                    mask = keep / (1.0 - self.p_drop)
                    a = a * mask
            else:
                a = z  # logits
            masks.append(mask)
        return inputs, masks, a

    @staticmethod
    def _log_softmax(logits):
        shifted = logits - logits.max(dim=1, keepdim=True).values
        return shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))

    def _backward(self, inputs, masks, logits, yb):
        """Manual backpropagation. Returns ``[(dW_0, db_0), ..., (dW_L, db_L)]``."""
        batch = yb.shape[0]
        onehot = torch.zeros_like(logits)
        onehot[torch.arange(batch), yb] = 1.0
        delta = (torch.exp(self._log_softmax(logits)) - onehot) / batch
        grads = [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            a_prev = inputs[i]
            grads[i] = (a_prev.T @ delta, delta.sum(dim=0))
            if i > 0:
                delta = (delta @ self.W[i].T) * (a_prev > 0).to(delta.dtype)
                if masks[i - 1] is not None:
                    delta = delta * masks[i - 1]
        return grads

    def loss_and_grads(self, X, y_idx, training=False):
        """Mean cross-entropy and manual gradients on a batch (``y_idx`` are class indices)."""
        X = torch.as_tensor(np.asarray(X), dtype=self._tdtype)
        y_idx = torch.as_tensor(np.asarray(y_idx), dtype=torch.long)
        inputs, masks, logits = self._forward_cache(X, training=training)
        loss = -self._log_softmax(logits)[torch.arange(len(y_idx)), y_idx].mean().item()
        return loss, self._backward(inputs, masks, logits, y_idx)

    def _eval_loss(self, X_t, y_t):
        total = 0.0
        for s in range(0, X_t.shape[0], 2048):
            _, _, logits = self._forward_cache(X_t[s : s + 2048], training=False)
            lp = self._log_softmax(logits)
            total += -lp[torch.arange(len(lp)), y_t[s : s + 2048]].sum().item()
        return total / X_t.shape[0]

    def _sgd_step(self, grads, velocity):
        """theta <- theta - lr * v,  v <- momentum * v + g  (identical to torch.optim.SGD)."""
        for i, (gW, gb) in enumerate(grads):
            vW, vb = velocity[i]
            vW.mul_(self.momentum).add_(gW)
            vb.mul_(self.momentum).add_(gb)
            self.W[i].sub_(self.learning_rate * vW)
            self.b[i].sub_(self.learning_rate * vb)

    def _snapshot(self):
        return [(W.clone(), b.clone()) for W, b in zip(self.W, self.b, strict=True)]

    def _restore(self, snap):
        self.W = [W.clone() for W, _ in snap]
        self.b = [b.clone() for _, b in snap]

    # ------------------------------------------------------------------ training
    def fit(self, X, y, X_val=None, y_val=None):
        """Train. ``X_val``/``y_val`` are only *monitored* (history); they never affect training."""
        X, y = check_Xy(X, y)
        self.classes, y_idx = np.unique(y, return_inverse=True)
        self.n_classes = len(self.classes)
        self._init_params(X.shape[1], self.n_classes)

        X_t = torch.from_numpy(X).to(self._tdtype)
        y_t = torch.from_numpy(y_idx).long()
        n = X_t.shape[0]

        n_stop = int(n * self.val_fraction) if self.early_stopping else 0
        if n_stop >= 1 and n - n_stop >= 1:
            order = torch.randperm(n, generator=self.generator)
            stop_idx, train_idx = order[:n_stop], order[n_stop:]
            X_stop, y_stop = X_t[stop_idx], y_t[stop_idx]
            X_tr, y_tr = X_t[train_idx], y_t[train_idx]
            self.stop_val_idx = stop_idx.numpy()
        else:
            n_stop = 0
            X_tr, y_tr = X_t, y_t
            self.stop_val_idx = None

        have_val = X_val is not None and y_val is not None
        if have_val:
            Xv = torch.from_numpy(check_X(X_val, X.shape[1], "X_val")).to(self._tdtype)
            yv = torch.from_numpy(np.searchsorted(self.classes, np.asarray(y_val))).long()

        velocity = [
            (torch.zeros_like(W), torch.zeros_like(b)) for W, b in zip(self.W, self.b, strict=True)
        ]
        self.history = {"loss": [], "inner_val_loss": [], "val_loss": [], "val_acc": []}
        best_loss, best_snap, wait = float("inf"), None, 0
        self.best_epoch_ = 0
        n_tr = X_tr.shape[0]

        for epoch in range(self.epochs):
            order = torch.randperm(n_tr, generator=self.generator)
            total = 0.0
            for s in range(0, n_tr, self.batch_size):
                idx = order[s : s + self.batch_size]
                loss, grads = self.loss_and_grads(X_tr[idx], y_tr[idx], training=True)
                total += loss * len(idx)
                self._sgd_step(grads, velocity)
            epoch_loss = total / n_tr
            if not np.isfinite(epoch_loss):
                raise FloatingPointError("Loss became NaN/inf - lower the learning rate.")
            self.history["loss"].append(epoch_loss)
            msg = f"epoch {epoch + 1}/{self.epochs} loss={epoch_loss:.4f}"

            if n_stop:
                inner = self._eval_loss(X_stop, y_stop)
                self.history["inner_val_loss"].append(inner)
                msg += f" inner_val_loss={inner:.4f}"
                if inner < best_loss - self.min_delta:
                    best_loss, best_snap, wait = inner, self._snapshot(), 0
                    self.best_epoch_ = epoch + 1
                else:
                    wait += 1
            if have_val:
                self.history["val_loss"].append(self._eval_loss(Xv, yv))
                logits = self._logits_t(Xv)
                self.history["val_acc"].append(float((logits.argmax(1) == yv).double().mean()))
                msg += f" val_acc={self.history['val_acc'][-1]:.4f}"
            if self.verbose:
                print(msg)
            if n_stop and wait >= self.patience:
                break

        self.stopped_epoch_ = len(self.history["loss"])
        if best_snap is not None:
            self._restore(best_snap)
            self.best_inner_loss_ = best_loss
        else:
            self.best_epoch_ = self.stopped_epoch_
            self.best_inner_loss_ = None
        if self.verbose:
            print(f"stopped at epoch {self.stopped_epoch_}, best epoch {self.best_epoch_}")
        return self

    # ------------------------------------------------------------------ inference
    def _logits_t(self, X_t):
        out = []
        for s in range(0, X_t.shape[0], 2048):
            out.append(self._forward_cache(X_t[s : s + 2048], training=False)[2])
        return torch.cat(out, dim=0)

    def _get_logits(self, X):
        self._check_fitted()
        X = check_X(X, self.W[0].shape[0])
        return self._logits_t(torch.from_numpy(X).to(self._tdtype))

    def predict(self, X):
        idx = self._get_logits(X).argmax(dim=1).numpy()
        return self.classes[idx]

    def predict_proba(self, X):
        return torch.exp(self._log_softmax(self._get_logits(X))).numpy().astype(np.float64)

    def predict_log_proba(self, X):
        return self._log_softmax(self._get_logits(X)).numpy().astype(np.float64)

    # ------------------------------------------------------------------ persistence
    def weights_numpy(self):
        """List of ``(W (fan_in, fan_out), b)`` float arrays (used by the ONNX export)."""
        self._check_fitted()
        return [(W.numpy().copy(), b.numpy().copy()) for W, b in zip(self.W, self.b, strict=True)]

    def _get_arrays(self):
        arrays = {"classes": np.asarray(self.classes)}
        for i, (W, b) in enumerate(self.weights_numpy()):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        return arrays

    def _set_arrays(self, arrays):
        self.classes = arrays["classes"]
        self.n_classes = len(self.classes)
        n_layers = len(self.hidden_sizes) + 1
        self.W = [
            torch.from_numpy(arrays[f"W{i}"].copy()).to(self._tdtype) for i in range(n_layers)
        ]
        self.b = [
            torch.from_numpy(arrays[f"b{i}"].copy()).to(self._tdtype) for i in range(n_layers)
        ]

    def _extra_state(self):
        return {
            "best_epoch": int(getattr(self, "best_epoch_", 0)),
            "stopped_epoch": int(getattr(self, "stopped_epoch_", 0)),
        }

    def _set_extra_state(self, extra):
        self.best_epoch_ = extra.get("best_epoch", 0)
        self.stopped_epoch_ = extra.get("stopped_epoch", 0)
