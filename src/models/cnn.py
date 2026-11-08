"""Small convolutional network with hand-written forward and backward passes.

Architecture (every size below is *derived* from the fitted model, see ``describe()``)::

    x (1, 28, 28) -> [Conv(k x k, valid) -> ReLU -> MaxPool(2 x 2)] * S -> Flatten
                  -> [Linear -> ReLU] * D -> Linear -> softmax

What is written by hand in this file (NumPy/torch tensors are used only as n-dimensional arrays):

* ``_im2col`` / ``_col2im``   sliding-window gather and its scatter-add adjoint (loops over the
  ``k x k`` kernel offsets; no ``conv2d`` and no ``unfold``-based convolution);
* convolution forward ``cols @ K`` and its gradients  ``dK = cols^T dY``, ``dcols = dY K^T``;
* max-pooling forward (window maximum and its arg-max) and backward (route the gradient to the
  arg-max position);
* ReLU, the dense layers, softmax cross-entropy, momentum SGD and early stopping.

Provided by PyTorch: ``torch.Tensor`` storage, ``@`` (matrix product), element-wise functions,
``torch.randn`` / ``torch.randperm`` for random numbers. *Autograd is never used for training* and
there is no ``nn.Module`` / ``torch.optim`` / ``F.conv2d`` object in the model.
``tests/test_cnn.py`` checks the convolution and pooling against a naive 7-loop reference and
against ``F.conv2d`` / ``F.max_pool2d``, and the gradients against finite differences and autograd.

The convolution is the deep-learning convention (cross-correlation, no kernel flip), identical to the
ONNX ``Conv`` operator used in the exported graph.
"""

import numpy as np
import torch

from src.models.base import BaseClassifier, NotFittedError, check_X, check_Xy

_DTYPES = {"float32": torch.float32, "float64": torch.float64}
IMAGE = (28, 28)


# ---------------------------------------------------------------------------- tensor primitives
def conv_out(size, k):
    return size - k + 1


def _im2col(x, k):
    """(B, C, H, W) -> (B*H'*W', C*k*k) patches, row-major over (b, y, x), columns over (c, i, j)."""
    B, C, H, W = x.shape
    Ho, Wo = conv_out(H, k), conv_out(W, k)
    cols = x.new_empty((B, Ho, Wo, C, k, k))
    for i in range(k):
        for j in range(k):
            cols[:, :, :, :, i, j] = x[:, :, i : i + Ho, j : j + Wo].permute(0, 2, 3, 1)
    return cols.reshape(B * Ho * Wo, C * k * k)


def _col2im(dcols, x_shape, k):
    """Adjoint of ``_im2col``: scatter-add patch gradients back to the image."""
    B, C, H, W = x_shape
    Ho, Wo = conv_out(H, k), conv_out(W, k)
    d = dcols.reshape(B, Ho, Wo, C, k, k)
    dx = dcols.new_zeros(x_shape)
    for i in range(k):
        for j in range(k):
            dx[:, :, i : i + Ho, j : j + Wo] += d[:, :, :, :, i, j].permute(0, 3, 1, 2)
    return dx


def conv_forward(x, K, b):
    """Valid cross-correlation. ``x`` (B,C,H,W), ``K`` (Cout,C,k,k), ``b`` (Cout,) -> (B,Cout,H',W')."""
    B, C, H, W = x.shape
    Cout, _, k, _ = K.shape
    Ho, Wo = conv_out(H, k), conv_out(W, k)
    cols = _im2col(x, k)
    y = cols @ K.reshape(Cout, -1).T + b
    return y.reshape(B, Ho, Wo, Cout).permute(0, 3, 1, 2).contiguous(), cols


def conv_backward(dy, cols, K, x_shape):
    """Gradients of a convolution: returns ``(dx, dK, db)``."""
    Cout, _, k, _ = K.shape
    dY = dy.permute(0, 2, 3, 1).reshape(-1, Cout)  # (B*H'*W', Cout)
    dK = (dY.T @ cols).reshape(K.shape)
    db = dY.sum(dim=0)
    dx = _col2im(dY @ K.reshape(Cout, -1), x_shape, k)
    return dx, dK, db


def maxpool_forward(x, p=2):
    """Non-overlapping ``p x p`` max pooling (odd borders are cropped). Returns ``(y, argmax)``."""
    B, C, H, W = x.shape
    Ho, Wo = H // p, W // p
    win = (
        x[:, :, : Ho * p, : Wo * p]
        .reshape(B, C, Ho, p, Wo, p)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(B, C, Ho, Wo, p * p)
    )
    y, arg = win.max(dim=-1)
    return y, arg


def maxpool_backward(dy, arg, x_shape, p=2):
    B, C, H, W = x_shape
    Ho, Wo = H // p, W // p
    win = dy.new_zeros((B, C, Ho, Wo, p * p))
    win.scatter_(-1, arg.unsqueeze(-1), dy.unsqueeze(-1))
    win = win.reshape(B, C, Ho, Wo, p, p).permute(0, 1, 2, 4, 3, 5).reshape(B, C, Ho * p, Wo * p)
    dx = dy.new_zeros(x_shape)
    dx[:, :, : Ho * p, : Wo * p] = win
    return dx


class CNN(BaseClassifier):
    def __init__(
        self,
        channels=(8, 16),
        *,
        kernel_size=3,
        pool=2,
        dense=(),
        learning_rate=0.03,
        momentum=0.9,
        p_drop=0.0,
        epochs=15,
        batch_size=128,
        seed=42,
        early_stopping=True,
        val_fraction=0.08,
        patience=3,
        min_delta=1e-4,
        dtype="float32",
        verbose=False,
    ):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        dense = tuple(int(d) for d in dense)
        if not channels or any(c < 1 for c in channels):
            raise ValueError("channels must be a non-empty tuple of positive integers.")
        if any(d < 1 for d in dense):
            raise ValueError("dense must contain positive integers.")
        if kernel_size < 1 or pool < 1:
            raise ValueError("kernel_size and pool must be >= 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if not 0.0 <= p_drop < 1.0:
            raise ValueError("p_drop must be in [0, 1).")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1.")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {list(_DTYPES)}.")
        # fail early when the image is too small for the requested stack
        h = IMAGE[0]
        for _ in channels:
            h = conv_out(h, kernel_size) // pool
            if h < 1:
                raise ValueError("channels / kernel_size / pool shrink the image below 1 pixel.")
        self.channels = channels
        self.kernel_size = int(kernel_size)
        self.pool = int(pool)
        self.dense = dense
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.p_drop = p_drop
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.early_stopping = early_stopping
        self.val_fraction = val_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.dtype = dtype
        self.verbose = verbose

    # ------------------------------------------------------------------ geometry / parameters
    @property
    def _tdtype(self):
        return _DTYPES[self.dtype]

    def _flat_size(self):
        h = IMAGE[0]
        for _ in self.channels:
            h = conv_out(h, self.kernel_size) // self.pool
        return self.channels[-1] * h * h

    def _init_params(self, n_classes):
        """He-normal kernels / dense weights, zero biases, one private seeded stream."""
        gen = torch.Generator().manual_seed(self.seed)

        def he(shape, fan_in):
            std = (2.0 / fan_in) ** 0.5
            return (torch.randn(*shape, generator=gen, dtype=torch.float64) * std).to(self._tdtype)

        self.K, self.kb = [], []
        c_in = 1
        for c in self.channels:
            self.K.append(
                he((c, c_in, self.kernel_size, self.kernel_size), c_in * self.kernel_size**2)
            )
            self.kb.append(torch.zeros(c, dtype=self._tdtype))
            c_in = c
        sizes = [self._flat_size(), *self.dense, n_classes]
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
            self.W.append(he((fan_in, fan_out), fan_in))
            self.b.append(torch.zeros(fan_out, dtype=self._tdtype))
        self.generator = gen

    def _check_fitted(self):
        if not hasattr(self, "K"):
            raise NotFittedError("CNN is not fitted yet.")

    # ------------------------------------------------------------------ forward / backward
    def _forward_cache(self, xb, training=False):
        """xb (B, 784) or (B,1,28,28). Returns ``(cache, logits)``; ``cache`` feeds ``_backward``."""
        x = xb.reshape(-1, 1, *IMAGE)
        cache = {"stages": [], "dense": []}
        for K, kb in zip(self.K, self.kb, strict=True):
            z, cols = conv_forward(x, K, kb)
            a = torch.relu(z)
            y, arg = maxpool_forward(a, self.pool)
            cache["stages"].append(
                {
                    "x_shape": tuple(x.shape),
                    "cols": cols,
                    "z": z,
                    "a_shape": tuple(a.shape),
                    "arg": arg,
                }
            )
            x = y
        a = x.reshape(x.shape[0], -1)
        cache["flat_shape"] = tuple(x.shape)
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            entry = {"input": a, "mask": None}
            z = a @ W + b
            if i < last:
                a = torch.relu(z)
                if training and self.p_drop > 0:
                    keep = (
                        torch.rand(a.shape, generator=self.generator, dtype=a.dtype) >= self.p_drop
                    ).to(a.dtype)
                    entry["mask"] = keep / (1.0 - self.p_drop)
                    a = a * entry["mask"]
            else:
                a = z
            cache["dense"].append(entry)
        return cache, a

    @staticmethod
    def _log_softmax(logits):
        shifted = logits - logits.max(dim=1, keepdim=True).values
        return shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))

    def _backward(self, cache, logits, yb):
        """Manual backpropagation. Returns ``(conv_grads, dense_grads)`` = lists of (dW, db)."""
        batch = yb.shape[0]
        onehot = torch.zeros_like(logits)
        onehot[torch.arange(batch), yb] = 1.0
        delta = (torch.exp(self._log_softmax(logits)) - onehot) / batch
        dense_grads = [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            a_prev = cache["dense"][i]["input"]
            dense_grads[i] = (a_prev.T @ delta, delta.sum(dim=0))
            delta = delta @ self.W[i].T
            if i > 0:
                delta = delta * (a_prev > 0).to(delta.dtype)
                if cache["dense"][i - 1]["mask"] is not None:
                    delta = delta * cache["dense"][i - 1]["mask"]
        d = delta.reshape(cache["flat_shape"])
        conv_grads = [None] * len(self.K)
        for s in range(len(self.K) - 1, -1, -1):
            st = cache["stages"][s]
            da = maxpool_backward(d, st["arg"], st["a_shape"], self.pool)
            dz = da * (st["z"] > 0).to(da.dtype)
            d, dK, db = conv_backward(dz, st["cols"], self.K[s], st["x_shape"])
            conv_grads[s] = (dK, db)
        return conv_grads, dense_grads

    def loss_and_grads(self, X, y_idx, training=False):
        """Mean cross-entropy and manual gradients: ``(loss, conv_grads, dense_grads)``."""
        X = torch.as_tensor(np.asarray(X), dtype=self._tdtype)
        y_idx = torch.as_tensor(np.asarray(y_idx), dtype=torch.long)
        cache, logits = self._forward_cache(X, training=training)
        loss = -self._log_softmax(logits)[torch.arange(len(y_idx)), y_idx].mean().item()
        conv_grads, dense_grads = self._backward(cache, logits, y_idx)
        return loss, conv_grads, dense_grads

    def _eval_loss(self, X_t, y_t):
        total = 0.0
        for s in range(0, X_t.shape[0], 1024):
            _, logits = self._forward_cache(X_t[s : s + 1024])
            lp = self._log_softmax(logits)
            total += -lp[torch.arange(len(lp)), y_t[s : s + 1024]].sum().item()
        return total / X_t.shape[0]

    def _params(self):
        return [*self.K, *self.kb, *self.W, *self.b]

    def _sgd_step(self, conv_grads, dense_grads, velocity):
        """``v <- momentum * v + g ;  theta <- theta - lr * v`` for every parameter tensor."""
        grads = [g[0] for g in conv_grads] + [g[1] for g in conv_grads]
        grads += [g[0] for g in dense_grads] + [g[1] for g in dense_grads]
        for p, g, v in zip(self._params(), grads, velocity, strict=True):
            v.mul_(self.momentum).add_(g)
            p.sub_(self.learning_rate * v)

    def _snapshot(self):
        return [p.clone() for p in self._params()]

    def _restore(self, snap):
        n_k, n_w = len(self.K), len(self.W)
        self.K = [t.clone() for t in snap[:n_k]]
        self.kb = [t.clone() for t in snap[n_k : 2 * n_k]]
        self.W = [t.clone() for t in snap[2 * n_k : 2 * n_k + n_w]]
        self.b = [t.clone() for t in snap[2 * n_k + n_w :]]

    # ------------------------------------------------------------------ training
    def fit(self, X, y, X_val=None, y_val=None, augment=None, callback=None):
        """Train. ``X_val``/``y_val`` are only *monitored*. ``augment(X_batch, epoch)`` (optional)
        returns a transformed copy of a training batch; it is never applied to validation data.
        ``callback(epoch, model)`` (optional, for training replays) is called with epoch 0 after
        initialisation and after every epoch; it must not modify the model."""
        X, y = check_Xy(X, y)
        if X.shape[1] != IMAGE[0] * IMAGE[1]:
            raise ValueError("CNN expects flattened 28x28 images (784 features).")
        self.classes, y_idx = np.unique(y, return_inverse=True)
        self.n_classes = len(self.classes)
        self._init_params(self.n_classes)

        X_t = torch.from_numpy(X).to(self._tdtype)
        y_t = torch.from_numpy(y_idx).long()
        n = X_t.shape[0]
        n_stop = int(n * self.val_fraction) if self.early_stopping else 0
        if n_stop >= 1 and n - n_stop >= 1:
            order = torch.randperm(n, generator=self.generator)
            stop_idx, train_idx = order[:n_stop], order[n_stop:]
            X_stop, y_stop = X_t[stop_idx], y_t[stop_idx]
            X_tr, y_tr = X_t[train_idx], y_t[train_idx]
        else:
            n_stop = 0
            X_tr, y_tr = X_t, y_t
        have_val = X_val is not None and y_val is not None
        if have_val:
            Xv = torch.from_numpy(check_X(X_val, X.shape[1], "X_val")).to(self._tdtype)
            yv = torch.from_numpy(np.searchsorted(self.classes, np.asarray(y_val))).long()

        velocity = [torch.zeros_like(p) for p in self._params()]
        self.history = {
            "loss": [],
            "inner_val_loss": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_time": [],
        }
        best_loss, best_snap, wait = float("inf"), None, 0
        self.best_epoch_ = 0
        n_tr = X_tr.shape[0]
        import time

        if callback is not None:
            callback(0, self)
        for epoch in range(self.epochs):
            t0 = time.perf_counter()
            order = torch.randperm(n_tr, generator=self.generator)
            total = 0.0
            for s in range(0, n_tr, self.batch_size):
                idx = order[s : s + self.batch_size]
                xb = X_tr[idx]
                if augment is not None:
                    xb = augment(xb, epoch)
                loss, cg, dg = self.loss_and_grads(xb, y_tr[idx], training=True)
                total += loss * len(idx)
                self._sgd_step(cg, dg, velocity)
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
                self.history["val_acc"].append(
                    float((self._logits_t(Xv).argmax(1) == yv).double().mean())
                )
                msg += f" val_acc={self.history['val_acc'][-1]:.4f}"
            self.history["epoch_time"].append(time.perf_counter() - t0)
            if self.verbose:
                print(msg)
            if callback is not None:
                callback(epoch + 1, self)
            if n_stop and wait >= self.patience:
                break

        self.stopped_epoch_ = len(self.history["loss"])
        if best_snap is not None:
            self._restore(best_snap)
            self.best_inner_loss_ = best_loss
        else:
            self.best_epoch_ = self.stopped_epoch_
            self.best_inner_loss_ = None
        return self

    # ------------------------------------------------------------------ inference / inspection
    def _logits_t(self, X_t):
        out = []
        for s in range(0, X_t.shape[0], 1024):
            out.append(self._forward_cache(X_t[s : s + 1024])[1])
        return torch.cat(out, dim=0)

    def _get_logits(self, X):
        self._check_fitted()
        X = check_X(X, IMAGE[0] * IMAGE[1])
        return self._logits_t(torch.from_numpy(X).to(self._tdtype))

    def predict(self, X):
        return self.classes[self._get_logits(X).argmax(dim=1).numpy()]

    def predict_proba(self, X):
        return torch.exp(self._log_softmax(self._get_logits(X))).numpy().astype(np.float64)

    def predict_log_proba(self, X):
        return self._log_softmax(self._get_logits(X)).numpy().astype(np.float64)

    def forward_trace(self, X):
        """Every named intermediate tensor for the rows of ``X`` (standardised, (N, 784)).

        Keys: ``input``, then per conv stage ``conv{s}`` (pre-activation), ``act{s}`` (ReLU),
        ``pool{s}``; then ``flat``, ``z{d}`` / ``a{d}`` for hidden dense layers and ``logits``.
        These are exactly the tensors the exported ONNX graph exposes as trace outputs.
        """
        self._check_fitted()
        X = check_X(X, IMAGE[0] * IMAGE[1])
        x = torch.from_numpy(X).to(self._tdtype).reshape(-1, 1, *IMAGE)
        out = {"input": x.numpy().copy()}
        for s, (K, kb) in enumerate(zip(self.K, self.kb, strict=True)):
            z, _ = conv_forward(x, K, kb)
            a = torch.relu(z)
            x, _ = maxpool_forward(a, self.pool)
            out[f"conv{s}"], out[f"act{s}"], out[f"pool{s}"] = z.numpy(), a.numpy(), x.numpy()
        a = x.reshape(x.shape[0], -1)
        out["flat"] = a.numpy()
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            z = a @ W + b
            if i < len(self.W) - 1:
                out[f"z{i}"] = z.numpy()
                a = torch.relu(z)
                out[f"a{i}"] = a.numpy()
            else:
                out["logits"] = z.numpy()
        return out

    def describe(self):
        """Layer list with shapes obtained by running the actual forward code on one blank image."""
        self._check_fitted()
        tr = self.forward_trace(np.zeros((1, IMAGE[0] * IMAGE[1])))
        layers = [{"id": "input", "op": "input", "out_shape": [1, *IMAGE]}]
        for s, K in enumerate(self.K):
            layers += [
                {
                    "id": f"conv{s}",
                    "op": "conv2d",
                    "kernel": list(K.shape),
                    "stride": 1,
                    "padding": 0,
                    "out_shape": list(tr[f"conv{s}"].shape[1:]),
                },
                {"id": f"act{s}", "op": "relu", "out_shape": list(tr[f"act{s}"].shape[1:])},
                {
                    "id": f"pool{s}",
                    "op": "maxpool2d",
                    "window": self.pool,
                    "out_shape": list(tr[f"pool{s}"].shape[1:]),
                },
            ]
        layers.append({"id": "flat", "op": "flatten", "out_shape": list(tr["flat"].shape[1:])})
        for i, W in enumerate(self.W):
            last = i == len(self.W) - 1
            layers.append(
                {
                    "id": "logits" if last else f"z{i}",
                    "op": "dense",
                    "weights": list(W.shape),
                    "out_shape": [W.shape[1]],
                }
            )
            if not last:
                layers.append({"id": f"a{i}", "op": "relu", "out_shape": [W.shape[1]]})
        return layers

    def n_parameters(self):
        self._check_fitted()
        return int(sum(p.numel() for p in self._params()))

    # ------------------------------------------------------------------ persistence
    def weights_numpy(self):
        """``{'K': [...], 'kb': [...], 'W': [...], 'b': [...]}`` copies as float arrays."""
        self._check_fitted()
        return {
            "K": [k.numpy().copy() for k in self.K],
            "kb": [k.numpy().copy() for k in self.kb],
            "W": [w.numpy().copy() for w in self.W],
            "b": [b.numpy().copy() for b in self.b],
        }

    def _get_arrays(self):
        arrays = {"classes": np.asarray(self.classes)}
        for name, lst in self.weights_numpy().items():
            for i, a in enumerate(lst):
                arrays[f"{name}{i}"] = a
        return arrays

    def _set_arrays(self, arrays):
        self.classes = arrays["classes"]
        self.n_classes = len(self.classes)
        t = lambda k: torch.from_numpy(arrays[k].copy()).to(self._tdtype)  # noqa: E731
        self.K = [t(f"K{i}") for i in range(len(self.channels))]
        self.kb = [t(f"kb{i}") for i in range(len(self.channels))]
        n_dense = len(self.dense) + 1
        self.W = [t(f"W{i}") for i in range(n_dense)]
        self.b = [t(f"b{i}") for i in range(n_dense)]

    def _extra_state(self):
        return {
            "best_epoch": int(getattr(self, "best_epoch_", 0)),
            "stopped_epoch": int(getattr(self, "stopped_epoch_", 0)),
        }

    def _set_extra_state(self, extra):
        self.best_epoch_ = extra.get("best_epoch", 0)
        self.stopped_epoch_ = extra.get("stopped_epoch", 0)
