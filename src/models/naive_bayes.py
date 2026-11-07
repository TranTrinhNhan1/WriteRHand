import numpy as np
from src.models.base import BaseClassifier

class GaussianNB(BaseClassifier):
    """Gaussian Naive Bayes: per-class, per-pixel normal likelihoods + class priors (MLE).

    ``var_smoothing`` adds ``var_smoothing * max_j Var(x_j)`` to every variance (same convention
    as scikit-learn) so constant pixels do not give a zero variance.
    """

    _state_arrays = ("classes", "mean", "var", "prior")

    def __init__(self, var_smoothing=1e-9):
        super().__init__()
        if var_smoothing < 0:
            raise ValueError("var_smoothing must be >= 0.")
        self.var_smoothing = var_smoothing

    def _set_arrays(self, arrays):
        super()._set_arrays(arrays)
        self.n_classes, self.n_features = self.mean.shape

    def _check_X(self, X):
        ''' check input shape and values '''
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("X must contain numeric values.")

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (N, D).")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must not be empty.")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or infinity.")

        return X

    def fit(self, X, y):
        X = self._check_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if len(y) == 0:
            raise ValueError("Training data must not be empty.")

        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.mean = np.zeros((self.n_classes, self.n_features))
        self.var = np.zeros((self.n_classes, self.n_features))
        self.prior = np.zeros(self.n_classes)
        eps = self.var_smoothing * X.var(axis=0).max() # avoid var = 0

        for idx, c in enumerate(self.classes):
            X_c = X[y==c]
            self.mean[idx] = X_c.mean(axis=0)
            self.var[idx] = X_c.var(axis=0) + eps
            self.prior[idx] = X_c.shape[0] / self.n_samples

        return self

    def _joint_log_likelihood(self, X):
        """
        Calculate joint log-likelihood for each sample and class.
        Returns: Array with shape (N, K).
        """
        X = self._check_X(X)

        if X.shape[1] != self.n_features:
            raise ValueError("X has a different number of features than training data.")

        n_samples = X.shape[0]
        n_classes = len(self.classes)

        # Calculate class constants once: shape (K,)
        constants = (
            np.log(self.prior) - 0.5 * np.sum(
                np.log(2 * np.pi * self.var), axis=1
            )
        )
        # Store scores: shape (N, K)
        scores = np.empty((n_samples, n_classes), dtype=np.float64)
        batch_size = 4096

        for k in range(n_classes):
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)

                # Calculate squared difference for class k
                d = X[start:end] - self.mean[k]
                np.square(d, out=d)
                d /= self.var[k]

                # Calculate scores for class k
                scores[start:end, k] = (constants[k] - 0.5 * np.sum(d, axis=1))

        return scores

    def predict(self, X):
        ''' pred class label '''
        joint_log_likelihood = self._joint_log_likelihood(X)
        best_class_indices = np.argmax(joint_log_likelihood, axis=1)
        return self.classes[best_class_indices]

    def predict_log_proba(self, X):
        ''' calc normalized log probabilities '''
        joint_log_likelihood = self._joint_log_likelihood(X)
        max_log = np.max(joint_log_likelihood, axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(
            np.sum(np.exp(joint_log_likelihood - max_log), axis=1, keepdims=True)
        )

        return joint_log_likelihood - log_sum_exp

    def predict_proba(self, X):
        """Calculate normalized probabilities."""
        return np.exp(self.predict_log_proba(X))


if __name__ == "__main__":
    
    from src.preprocessing.preprocess import load_mnist
    from src.models.cross_validation import *

    Xtr_raw, ytr_raw, _, _ = load_mnist()
    X_train, y_train = Xtr_raw[:5000], ytr_raw[:5000]

    # Khai báo lưới siêu tham số
    param_grid = {
        "var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0]
    }
    best_params, results = find_grid(
        model_class=GaussianNB,
        param_grid=param_grid,
        X=X_train,
        y=y_train,
        K=4,
        seed=SEED,
        scoring="accuracy",
        labels=range(10)
    )

    # In kết quả
    print("\nCấu hình tốt nhất:")
    print(best_params)

    print("\nKết quả từng cấu hình:")
    for result in results:
        print(
            f"Params: {result['params']}, "
            f"Accuracy: {result['mean_scores']['accuracy']:.4f}, "
            f"Std: {result['std_scores']['accuracy']:.4f}"
        )