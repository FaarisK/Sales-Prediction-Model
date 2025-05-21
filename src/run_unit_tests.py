import numpy as np
from src.my_decision_tree import MyDecisionTree
from src.my_logistic_regression import MyLogisticRegression

# tiny XOR‑like dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])             # only 1 when both features =1

tree = MyDecisionTree(max_depth=2)
tree.fit(X, y)
print("Tree predicts:", tree.predict(X), "\n")

lr = MyLogisticRegression(lr=0.5, n_iter=3000, reg=0.0)
lr.fit(X, y)
print("LogReg predicts:", lr.predict(X))