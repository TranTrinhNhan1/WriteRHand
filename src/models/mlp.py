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
    def __init__(self, hidden_sizes=(128,), *,
                 learning_rate=0.1, p_drop = 0.0,
                 epochs=20, batch_size=128,
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
        if not hasattr(self, "layers_"):
            raise RuntimeError("Model is not fitted yet.")

    @staticmethod
    def _softmax(z):
        z = z - max(axis=1, keepdims=True) # tranh tran so
        # tinh theo hang va giu dim dang (N, 1)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True) 

    @staticmethod
    def _log_softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

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
        return torch.optim.sgd(params, lr=self.learning_rate, momentum=self.momentum)

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
        self.history = {"loss": [], "val_acc": []}

        with torch.no_grad():
            for epoch in range(self.epochs):
                epoch_loss = self._train_one_epoch(X_t, y_t, optimizer)
                if not np.isfinite(epoch_loss):
                    raise FloatingPointError("Loss is Nan/Inf")
                self.history["loss"].append(epoch_loss)

            has_val = False
            if X_val is not None and y_val is not None:
                y_val = np.asarray(y_val)
                val_acc = np.mean(self.predict(X_val)) == y_val # dem so lan dung va lay trung binh
                has_val = True
            if self.verbose:
                verbose_line = f"epoch {epoch+1}/{self.epochs}  loss={epoch_loss:.3f}"
                if has_val:
                    verbose_line += f"  val_acc={val_acc:.3f}"
                print(verbose_line)

        return self

    # CAC HAM PREDICT
    def _get_logits(self, X):
        self._check_fitted()
        X = self._check_X(X)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                chunk = torch.from_numpy(X[start:start + self.batch_size]).float()
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