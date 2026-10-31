import numpy as np
from src.models.base import BaseClassifier

class GaussianNB(BaseClassifier):
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

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

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.prior = np.zeros(n_classes)
        eps = self.var_smoothing * X.var(axis=0).max() # avoid var = 0

        for idx, c in enumerate(self.classes):
            X_c = X[y==c]
            self.mean[idx] = X_c.mean(axis=0)
            self.var[idx] = X_c.var(axis=0) + eps
            self.prior[idx] = X_c.shape[0] / n_samples

        return self

    def _joint_log_likelihood(self, X):
        """
        Calculate joint log-likelihood for each sample and class.
        Returns: Array with shape (N, K).
        """
        X = self._check_X(X)
        # Calculate log Gaussian probability
        log_probability = np.log(2 * np.pi * self.var)
        squared_difference = (
            X[:, np.newaxis, :] - self.mean[np.newaxis, :, :]
        ) ** 2
        gaussian_term = np.sum(
            log_probability[np.newaxis, :, :]
            + squared_difference / self.var[np.newaxis, :, :], axis=2
        )

        # Add log class prior
        joint_log_likelihood = (
            np.log(self.prior)[np.newaxis, :]
            - 0.5 * gaussian_term
        )
        return joint_log_likelihood

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
    from src.models.cross_validation import find_grid, SEED

    Xtr_raw, ytr_raw, _, _ = load_mnist()
    X_train, y_train = Xtr_raw[:5000], ytr_raw[:5000]

    # Khai báo lưới siêu tham số
    param_grid = {
        "var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]
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