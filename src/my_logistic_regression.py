"""
my_logistic_regression.py
Binary logistic regression (batch gradient descent, L2 reg).
Author: Faaris Khan
"""

import numpy as np

class MyLogisticRegression:
    def __init__(self, lr=0.1, n_iter=2000, reg=0.0, random_state=42):
        self.lr, self.n_iter, self.reg = lr, n_iter, reg
        self.random_state = random_state
        self.w_ = None

    # ---------- helpers ----------
    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -20, 20)      # numerical stability
        return 1.0 / (1.0 + np.exp(-z))

    # ---------- API ----------
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        Xb  = np.c_[np.ones((X.shape[0], 1)), X]     # add bias
        self.w_ = rng.normal(scale=0.01, size=Xb.shape[1])

        for _ in range(self.n_iter):
            probs = self._sigmoid(Xb @ self.w_)
            grad  = Xb.T @ (probs - y) / len(y) + self.reg * self.w_
            self.w_ -= self.lr * grad
        return self                                  # <‑‑ return self

    def predict_proba(self, X):
        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        return self._sigmoid(Xb @ self.w_)

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X) >= thr).astype(int)
