from src.models.base import BaseClassifier
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


'''
hyper_param can fine-tune
batch_size
hidden_sizes
p_drop
learning_rate
momentum
'''

class MLP(BaseClassifier):
    VAL_FRACTION = 0.08
    PATIENCE = 5
    MIN_DELTA = 1e-4

    def __init__(self, hidden_sizes=(128,), *,
                 learning_rate=0.1, p_drop = 0.0,
                 epochs=100, batch_size=128,
                 momentum=0.0, seed=42, verbose=False):
        self.hidden_sizes = tuple(hidden_sizes)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.p_drop = p_drop
        self.seed = seed
        self.verbose = verbose

    def _check_X(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array!")
        if not np.isfinite(X).all():
            raise ValueError("X cannot contain Nan or inf")
        return X 

    def _check_fitted(self):
        if not hasattr(self, "layers"):
            raise RuntimeError("Model is not fitted yet.")

    # CAC HAM CHO EARLY STOPPING

    def _split_internal(self, n_samples):
        ''' tach mot phan nho cua tap train de danh gia loss, phuc vu cho early stopping
         tra ve index cua tap train thuc su va tap val cho early stopping '''
        order = torch.randperm(n_samples, generator=self.generator)
        n_val = int(n_samples * self.VAL_FRACTION) # lay VAL_FRACTION phan tram cua tap train\
        if n_val < 1 or n_samples - n_val < 1:
            return order, None
        return order[n_val:], order[:n_val] 

    def _eval_loss(self, X_t, y_t):
        total = 0.0
        with torch.no_grad():
            for start in range(0, X_t.shape[0], 1024):
                xb = X_t[start:start+1024]
                yb = y_t[start:start+1024]
                _, _, logits = self._forward_cache(xb, training=False)
                total += F.cross_entropy(logits, yb, reduction="sum").item() # tra ve tong loss cua batch
        return total / X_t.shape[0]

    def _snapshot(self):
        # tach khoi ma tran autograd roi sao chep sang vung nho moi, de khi cap nhat trong so thi trong so duoc luu khong thay doi theo
        return [
            (lin.weight.detach().clone(), lin.bias.detach().clone())
            for lin in self.layers
        ]

    def _restore_best(self, snapshot):
        with torch.no_grad():
            for lin, (w, b) in zip(self.layers, snapshot):
                lin.weight.copy_(w)
                lin.bias.copy_(b)

    # CAC HAM CHO FIT

    def _build_layers(self, n_feat, n_classes):
        ''' dung torch tao kien truc mang voi cac node va bias ban dau
        tra ve layers chua cac lop, moi lop gom cac node '''
        sizes = [n_feat, *self.hidden_sizes, n_classes]
        layers = []
        for i in range(len(sizes) - 1):
            lin = nn.Linear(sizes[i], sizes[i+1])
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu") # khoi tao He
            nn.init.zeros_(lin.bias) # khoi tao bias bang 0
            layers.append(lin)
        return layers

    def _forward_cache(self, xb, training=False):
        ''' training: chi True khi train, con False khi predict, vi pred khong can dropout

        '''
        inputs = []
        masks = []
        a = xb
        last = len(self.layers) - 1 # lop cuoi khong co dropout
        for i, lin in enumerate(self.layers):
            inputs.append(a)
            z = lin(a)
            if i < last:
                a = torch.relu(z)
                if training and self.p_drop > 0:
                    rand = torch.rand(a.shape, generator=self.generator)
                    keep = (rand >= self.p_drop).float()
                    mask = keep / (1.0 - self.p_drop)
                    a = a * mask # logits
                else:
                    mask = None
            else:
                a = z # logits, dau vao cua lop tiep theo
                mask = None
            masks.append(mask)
        return inputs, masks, a

    def _backward(self, inputs, masks, logits, yb):
        ''' chi tinh va luu dao ham, chua cap nhat tham so, khong co tra ve '''
        batch = yb.shape[0]
        onehot = F.one_hot(yb, self.n_classes).float()
        delta = (torch.softmax(logits, dim=1) - onehot) / batch

        for i in range(len(self.layers)-1, -1, -1): # duyet tu lop cuoi
            lin = self.layers[i]
            a_prev = inputs[i]
            lin.weight.grad = delta.T @ a_prev
            lin.bias.grad = delta.sum(dim=0)
            if i > 0:
                delta = (delta @ lin.weight) * (a_prev > 0)
                if masks[i - 1] is not None:
                    delta = delta * masks[i - 1]

    def _make_optimizer(self):
        ''' cap nhat trong so '''
        params = [p for lin in self.layers for p in lin.parameters()]
        return torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum)

    def _train_one_epoch(self, X_tr, y_tr, optimizer):
        n_samples = X_tr.shape[0]
        order = torch.randperm(n_samples, generator=self.generator) # hoan vi de cac batch train theo thu tu khac nhau
        total_loss = 0.0

        for start in range(0, n_samples, self.batch_size):
            idx = order[start:start + self.batch_size]
            xb = X_tr[idx]
            yb = y_tr[idx]

            inputs, masks, logits = self._forward_cache(xb, training=True)
            total_loss += F.cross_entropy(logits, yb).item() * len(idx) # cong loss theo kich thuoc batch
            self._backward(inputs, masks, logits, yb)
            optimizer.step()
        return total_loss / n_samples

    def fit(self, X, y, X_val=None, y_val=None):
        X = self._check_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1D and have the same length with X")
        self.classes, y_idx = np.unique(y, return_inverse=True) # self.classes chua cac label da sort, y_idx chua mang chi so cua tung label trong self.classes
        self.n_classes = len(self.classes)

        torch.manual_seed(self.seed)
        self.generator = torch.Generator().manual_seed(self.seed)

        self.layers = self._build_layers(X.shape[1], self.n_classes)
        optimizer = self._make_optimizer()

        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y_idx).long()
        # chia tap train de early stopping
        train_idx, stop_val_idx = self._split_internal(X_t.shape[0])
        X_train, y_train = X_t[train_idx], y_t[train_idx]
        if stop_val_idx is not None:
            X_stop, y_stop = X_t[stop_val_idx], y_t[stop_val_idx]
            self.stop_val_idx = stop_val_idx.numpy()
        else:
            self.stop_val_idx = None

        self.history = {"loss": [], "inner_val_loss": [], "val_acc": []}
        best_loss = float("inf")
        best_snapshot = None
        self.best_epoch = 0

        # Training
        with torch.no_grad():
            for epoch in range(self.epochs):
                epoch_loss = self._train_one_epoch(X_train, y_train, optimizer)
                if not np.isfinite(epoch_loss):
                    raise FloatingPointError(
                        "Loss became NaN/inf!")
                self.history["loss"].append(epoch_loss)
                message = f"epoch {epoch + 1}/{self.epochs} loss={epoch_loss:.4f}"

                if stop_val_idx is not None:
                    inner_loss = self._eval_loss(X_stop, y_stop)
                    self.history["inner_val_loss"].append(inner_loss)
                    message += f" inner_val_loss={inner_loss:.4f}"
                    if inner_loss < best_loss - self.MIN_DELTA:
                        best_loss = inner_loss
                        best_snapshot = self._snapshot()
                        self.best_epoch_ = epoch + 1
                        wait = 0
                    else:
                        wait += 1

                if X_val is not None and y_val is not None:
                    val_acc = np.mean(self.predict(X_val) == np.asarray(y_val))
                    self.history["val_acc"].append(val_acc)
                    message += f" val_acc={val_acc:.4f}"
                if self.verbose:
                    print(message)

                if stop_val_idx is not None and wait >= self.PATIENCE:
                    break

            self.stopped_epoch_ = len(self.history["loss"])
            if best_snapshot is not None:
                self._restore_best(best_snapshot)
                self.best_inner_loss_ = best_loss
            else:
                self.best_epoch_ = self.stopped_epoch_
                self.best_inner_loss_ = None

        if self.verbose:
            print(f"stopped at epoch {self.stopped_epoch_}, "
                  f"best epoch {self.best_epoch_}")

        return self

    # CAC HAM PREDICT
    def _get_logits(self, X):
        self._check_fitted()
        X = self._check_X(X)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), 1024): # lay tung batch de khi forward khong tran ram
                chunk = torch.from_numpy(X[start:start + 1024]).float()
                _, _, logits = self._forward_cache(chunk, training=False)
                outputs.append(logits)
        return torch.cat(outputs, dim=0)  # noi theo hang

    def predict(self, X):
        idx = self._get_logits(X).argmax(dim=1).numpy()
        return self.classes[idx]

    def predict_proba(self, X):
        return torch.softmax(self._get_logits(X), dim=1).numpy().astype(np.float64)

    def predict_log_proba(self, X):
        return torch.log_softmax(self._get_logits(X), dim=1).numpy().astype(np.float64)