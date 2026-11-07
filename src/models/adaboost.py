"""Multiclass AdaBoost (SAMME, Zhu et al. 2009) with from-scratch weighted decision trees.

Weak learner  ``WeightedTree``: depth-limited CART classifier minimising the *weighted Gini
              impurity*. Features are quantised once into ``n_bins`` uniform bins so that the best
              split of every node can be found from weighted class histograms (O(N*D) per level).
Booster       ``AdaBoostSAMME``:
                  w_i <- 1/N
                  for t = 1..T:   h_t = tree fitted with weights w
                                  err = sum_i w_i [h_t(x_i) != y_i]            (w normalised)
                                  a_t = lr * ( ln((1-err)/err) + ln(K-1) )
                                  w_i <- w_i * exp(a_t [h_t(x_i) != y_i]);  renormalise
                  H(x) = argmax_k  sum_t a_t [h_t(x) = k]
Only NumPy is used; scikit-learn appears in the tests as an independent reference.
"""

import numpy as np

from src.models.base import BaseClassifier, check_X, check_Xy


class WeightedTree:
    """Depth-limited classification tree on binned features with sample weights (Gini)."""

    def __init__(self, max_depth=3, n_bins=32):
        self.max_depth = max_depth
        self.n_bins = n_bins

    def fit_binned(self, Xb_f, y_idx, w, n_classes, feature_ids, lo, step):
        """Fit on column-major uint8 bins ``Xb_f`` (N, D). ``feature_ids``: candidate features."""
        n = Xb_f.shape[0]
        B, K = self.n_bins, n_classes
        cap = 2 ** (self.max_depth + 1) - 1
        feature = np.full(cap, -1, dtype=np.int32)
        threshold = np.zeros(cap)
        left = np.full(cap, -1, dtype=np.int32)
        right = np.full(cap, -1, dtype=np.int32)
        value = np.zeros((cap, K))
        node_of = np.zeros(n, dtype=np.int64)  # node id of each sample, -1 once it sits in a leaf
        n_nodes = 1
        active = [0]
        value[0] = np.bincount(y_idx, weights=w, minlength=K)

        for _depth in range(self.max_depth):
            if not active:
                break
            M = len(active)
            local = np.full(cap, M, dtype=np.int64)  # node id -> slot (M = "inactive" slot)
            local[active] = np.arange(M)
            slot = np.where(node_of >= 0, local[np.maximum(node_of, 0)], M)
            L = slot * K + y_idx
            MK = (M + 1) * K
            F = len(feature_ids)
            H = np.empty((F, B, M, K))
            for j, f in enumerate(feature_ids):
                h = np.bincount(Xb_f[:, f].astype(np.intp) * MK + L, weights=w, minlength=B * MK)
                H[j] = h.reshape(B, M + 1, K)[:, :M, :]

            left_c = np.cumsum(H, axis=1)  # class weight with bin <= b
            tot = left_c[:, -1:, :, :]
            right_c = tot - left_c
            Wl, Wr = left_c.sum(-1), right_c.sum(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                cost = (Wl - (left_c ** 2).sum(-1) / Wl) + (Wr - (right_c ** 2).sum(-1) / Wr)
            cost[(Wl <= 1e-15) | (Wr <= 1e-15)] = np.inf  # one side empty -> invalid split
            parent_w = tot[0, 0].sum(-1)  # (M,)
            parent_cost = parent_w - (tot[0, 0] ** 2).sum(-1) / np.maximum(parent_w, 1e-300)

            next_active = []
            split_nodes = {}
            for m, node in enumerate(active):
                c = cost[:, :, m]
                j, b = np.unravel_index(np.argmin(c), c.shape)
                if not np.isfinite(c[j, b]) or c[j, b] >= parent_cost[m] - 1e-12:
                    continue  # no useful split -> stays a leaf
                f = int(feature_ids[j])
                feature[node] = f
                threshold[node] = lo[f] + (b + 1) * step[f]
                l_id, r_id = n_nodes, n_nodes + 1
                n_nodes += 2
                left[node], right[node] = l_id, r_id
                value[l_id] = left_c[j, b, m]
                value[r_id] = right_c[j, b, m]
                split_nodes[node] = (f, int(b), l_id, r_id)
                next_active += [l_id, r_id]
            if not split_nodes:
                break
            # route samples
            new_node = node_of.copy()
            for node, (f, b, l_id, r_id) in split_nodes.items():
                idx = np.flatnonzero(node_of == node)
                go_left = Xb_f[idx, f] <= b
                new_node[idx[go_left]] = l_id
                new_node[idx[~go_left]] = r_id
            # samples whose node did not split become inactive (-1)
            still = np.isin(new_node, next_active)
            node_of = np.where(still, new_node, -1)
            active = next_active

        self.n_nodes = n_nodes
        self.feature, self.threshold = feature, threshold
        self.left, self.right = left, right
        self.value = value
        self.leaf_class = value.argmax(axis=1).astype(np.int32)
        return self

    def predict_idx(self, X):
        """Class *index* predicted for each row of the real-valued matrix ``X``."""
        node = np.zeros(len(X), dtype=np.int64)
        rows = np.arange(len(X))
        for _ in range(self.max_depth):
            f = self.feature[node]
            internal = f >= 0
            if not internal.any():
                break
            go_left = X[rows, np.maximum(f, 0)] < self.threshold[node]
            nxt = np.where(go_left, self.left[node], self.right[node])
            node = np.where(internal, nxt, node)
        return self.leaf_class[node]


class AdaBoostSAMME(BaseClassifier):
    """AdaBoost-SAMME with histogram-based weighted decision trees.

    Parameters
    ----------
    n_estimators : maximum number of boosting rounds.
    max_depth : depth of each weak tree (1 = decision stump).
    learning_rate : shrinkage applied to every estimator weight.
    n_bins : number of uniform feature bins used to search splits.
    max_features : fraction (0, 1] of features examined by each tree (random subspace), 1 = all.
    """

    _state_arrays = ("classes", "tree_feature", "tree_threshold", "tree_left", "tree_right",
                     "tree_leaf", "alphas")

    def __init__(self, n_estimators=100, max_depth=3, learning_rate=1.0, n_bins=32,
                 max_features=1.0, seed=42, verbose=False):
        super().__init__()
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if not 2 <= n_bins <= 256:
            raise ValueError("n_bins must be in [2, 256].")
        if not 0 < max_features <= 1:
            raise ValueError("max_features must be in (0, 1].")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_bins = n_bins
        self.max_features = max_features
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------ training
    def _bin(self, X, lo, step):
        Xb = np.empty(X.shape, dtype=np.uint8, order="F")
        for s in range(0, len(X), 8192):
            q = np.floor((X[s:s + 8192] - lo) / step)
            Xb[s:s + 8192] = np.clip(q, 0, self.n_bins - 1)
        return Xb

    def fit(self, X, y, X_val=None, y_val=None):
        """Train. ``X_val``/``y_val`` are only used to record a validation-accuracy curve."""
        X, y = check_Xy(X, y)
        self.classes, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes)
        if K < 2:
            raise ValueError("Need at least two classes.")
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        lo, hi = X.min(axis=0), X.max(axis=0)
        step = (hi - lo) / self.n_bins
        step[step == 0] = 1.0
        Xb = self._bin(X, lo, step)
        usable = np.flatnonzero(hi > lo)  # constant pixels can never be split on
        n_feat = max(1, int(round(self.max_features * len(usable))))
        have_val = X_val is not None and y_val is not None
        if have_val:
            X_val = check_X(X_val, d, "X_val")
            yv_idx = np.searchsorted(self.classes, np.asarray(y_val))
            score_val = np.zeros((len(X_val), K))
        score_tr = np.zeros((n, K))
        w = np.full(n, 1.0 / n)
        trees, alphas = [], []
        self.history = {"train_err_weighted": [], "alpha": [], "train_acc": [], "val_acc": []}

        for t in range(self.n_estimators):
            fids = (np.sort(rng.choice(usable, n_feat, replace=False))
                    if n_feat < len(usable) else usable)
            tree = WeightedTree(self.max_depth, self.n_bins).fit_binned(
                Xb, y_idx, w, K, fids, lo, step)
            pred = tree.predict_idx(X)
            miss = pred != y_idx
            err = float(w[miss].sum() / w.sum())
            if err >= 1.0 - 1.0 / K:  # no better than random guessing -> stop
                if not trees:
                    trees.append(tree)
                    alphas.append(1e-10)
                break
            err = max(err, 1e-12)
            alpha = self.learning_rate * (np.log((1.0 - err) / err) + np.log(K - 1.0))
            trees.append(tree)
            alphas.append(alpha)
            score_tr[np.arange(n), pred] += alpha
            self.history["train_err_weighted"].append(err)
            self.history["alpha"].append(float(alpha))
            self.history["train_acc"].append(float(np.mean(score_tr.argmax(1) == y_idx)))
            if have_val:
                score_val[np.arange(len(X_val)), tree.predict_idx(X_val)] += alpha
                self.history["val_acc"].append(float(np.mean(score_val.argmax(1) == yv_idx)))
            if self.verbose and (t + 1) % 10 == 0:
                print(f"round {t + 1}/{self.n_estimators} err={err:.4f} "
                      f"train_acc={self.history['train_acc'][-1]:.4f}")
            if err <= 1e-12:  # perfect weak learner: nothing left to boost
                break
            w = w * np.exp(alpha * miss)
            w /= w.sum()

        self._pack(trees, alphas)
        return self

    def _pack(self, trees, alphas):
        T = len(trees)
        cap = 2 ** (self.max_depth + 1) - 1
        self.tree_feature = np.full((T, cap), -1, dtype=np.int32)
        self.tree_threshold = np.zeros((T, cap))
        self.tree_left = np.full((T, cap), -1, dtype=np.int32)
        self.tree_right = np.full((T, cap), -1, dtype=np.int32)
        self.tree_leaf = np.zeros((T, cap), dtype=np.int32)
        for i, tr in enumerate(trees):
            self.tree_feature[i] = tr.feature
            self.tree_threshold[i] = tr.threshold
            self.tree_left[i] = tr.left
            self.tree_right[i] = tr.right
            self.tree_leaf[i] = tr.leaf_class
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.n_estimators_ = T
        self.n_features = None

    # ------------------------------------------------------------------ inference
    def _tree_predict(self, t, X):
        tr = WeightedTree(self.max_depth, self.n_bins)
        tr.feature, tr.threshold = self.tree_feature[t], self.tree_threshold[t]
        tr.left, tr.right, tr.leaf_class = self.tree_left[t], self.tree_right[t], self.tree_leaf[t]
        return tr.predict_idx(X)

    def decision_scores(self, X, n_estimators=None):
        """Weighted votes ``S_k(x) = sum_t a_t [h_t(x) = k]``, shape (N, K)."""
        self._check_fitted()
        X = check_X(X)
        T = self.n_estimators_ if n_estimators is None else min(n_estimators, self.n_estimators_)
        S = np.zeros((len(X), len(self.classes)))
        rows = np.arange(len(X))
        for t in range(T):
            S[rows, self._tree_predict(t, X)] += self.alphas[t]
        return S

    def staged_accuracy(self, X, y, steps):
        """Accuracy of the ensemble truncated to each size in ``steps`` (one pass over trees)."""
        self._check_fitted()
        X = check_X(X)
        y_idx = np.searchsorted(self.classes, np.asarray(y))
        S = np.zeros((len(X), len(self.classes)))
        rows = np.arange(len(X))
        out, steps = {}, sorted(set(int(s) for s in steps))
        for t in range(self.n_estimators_):
            S[rows, self._tree_predict(t, X)] += self.alphas[t]
            if (t + 1) in steps:
                out[t + 1] = float(np.mean(S.argmax(1) == y_idx))
        return out

    def predict(self, X):
        return self.classes[self.decision_scores(X).argmax(axis=1)]

    def predict_proba(self, X):
        """Normalised weighted vote shares (sum to 1). Not calibrated probabilities."""
        S = self.decision_scores(X)
        return S / S.sum(axis=1, keepdims=True)

    def _extra_state(self):
        return {"n_estimators_fitted": int(self.n_estimators_)}

    def _set_arrays(self, arrays):
        super()._set_arrays(arrays)
        self.n_estimators_ = len(self.alphas)
