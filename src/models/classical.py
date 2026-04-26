import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    """Логистическая регрессия с SGD"""
    def __init__(self, lr=0.01, max_iter=1000, tol=1e-4, batch_size=64, method="sgd", seed=42):
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.batch_size = int(batch_size)
        self.method = method
        self.seed = int(seed)
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _compute_loss(self, y_true, y_pred):
        eps = 1e-15
        return float(-np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)))

    def fit(self, X, y):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        self.coef_ = np.random.randn(n_features) * 0.01
        self.intercept_ = 0.0
        self.loss_history_ = []

        indices = np.arange(n_samples)
        for epoch in range(self.max_iter):
            np.random.shuffle(indices)
            start = 0
            while start < n_samples:
                end = min(start + self.batch_size, n_samples)
                idx = indices[start:end]
                X_batch, y_batch = X[idx], y[idx]

                z = X_batch @ self.coef_ + self.intercept_
                y_pred = self._sigmoid(z)
                err = y_pred - y_batch

                self.coef_ -= self.lr * (X_batch.T @ err) / len(idx)
                self.intercept_ -= self.lr * np.mean(err)
                start = end

            z_full = X @ self.coef_ + self.intercept_
            loss = self._compute_loss(y, self._sigmoid(z_full))
            self.loss_history_.append(loss)
            
            if len(self.loss_history_) > 1 and abs(self.loss_history_[-2] - loss) < self.tol:
                break
        return self

    def predict_proba(self, X):
        p = self._sigmoid(X @ self.coef_ + self.intercept_)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)