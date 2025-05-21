"""
my_decision_tree.py
Hand‑coded CART‑style decision tree for binary labels.
– Gini impurity
– Greedy best‑split search
– max_depth & min_samples stopping rules
Author: Faaris Khan
"""

import numpy as np
from collections import Counter

class MyDecisionTree:
    def __init__(self, max_depth=3, min_samples=2, random_state=42):
        self.max_depth    = max_depth
        self.min_samples  = min_samples
        self.random_state = random_state
        self.tree_        = None

    # -------- public API --------
    def fit(self, X, y):
        """X (n, d) ndarray of **numeric** features, y (n,) binary ints."""
        rng = np.random.default_rng(self.random_state)
        # guard against accidental string cols
        X = X.astype(float, copy=False)
        self.tree_ = self._grow(X, y, depth=0, rng=rng)
        return self                      # <‑‑  so callers can chain .predict()

    def predict(self, X):
        X = X.astype(float, copy=False)
        return np.array([self._walk(row, self.tree_) for row in X])

    # -------- helpers --------
    def _gini(self, y):
        counts = np.bincount(y, minlength=2)
        p = counts / len(y)
        return 1.0 - np.sum(p**2)

    def _best_split(self, X, y, rng):
        best_gini, best_feat, best_thr = 1.0, None, None
        n_samples, n_feats = X.shape

        for feat in range(n_feats):
            values = np.unique(X[:, feat])
            if len(values) == 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0   # mid‑points
            rng.shuffle(thresholds)
            for thr in thresholds:
                mask = X[:, feat] <= thr
                left, right = y[mask], y[~mask]
                if len(left) < self.min_samples or len(right) < self.min_samples:
                    continue
                g = (len(left)*self._gini(left) + len(right)*self._gini(right)) / n_samples
                if g < best_gini:
                    best_gini, best_feat, best_thr = g, feat, thr
        return best_feat, best_thr

    def _grow(self, X, y, depth, rng):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return {"leaf": int(Counter(y).most_common(1)[0][0])}

        feat, thr = self._best_split(X, y, rng)
        if feat is None:
            return {"leaf": int(Counter(y).most_common(1)[0][0])}

        mask = X[:, feat] <= thr
        return {
            "feat": int(feat),
            "thr":  float(thr),
            "left":  self._grow(X[mask],  y[mask],  depth+1, rng),
            "right": self._grow(X[~mask], y[~mask], depth+1, rng)
        }

    def _walk(self, row, node):
        if "leaf" in node:
            return node["leaf"]
        branch = node["left"] if row[node["feat"]] <= node["thr"] else node["right"]
        return self._walk(row, branch)
