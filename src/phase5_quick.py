"""
phase5_quick.py  – robust version
• Drops string cols (date)
• Uses 3‑fold CV to avoid too‑small folds
• Writes cv_table.csv + best_params.json
"""

import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from my_decision_tree import MyDecisionTree
from my_logistic_regression import MyLogisticRegression

print(">>> phase‑5 quick grid‑search")

ROOT = Path(__file__).resolve().parents[1]
fig_dir  = ROOT / "figures"; fig_dir.mkdir(exist_ok=True)
data_dir = ROOT / "data";    data_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_dir / "train.csv")

# --- drop string columns ---
X = df.drop(columns=["high_demand", "date"]).to_numpy(dtype=float)
y = df["high_demand"].to_numpy()
print("train shape:", X.shape)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def safe_cv_f1(model_factory):
    scores = []
    for tr,va in cv.split(X, y):
        try:
            mdl = model_factory().fit(X[tr], y[tr])
            scores.append(f1_score(y[va], mdl.predict(X[va])))
        except Exception as e:
            print("  ❗️ model failed:", e)
            return -1  # ensure it won't be selected
    return float(np.mean(scores))

# --- grid search ---
depths  = [2,3,4,5,6]
lambdas = [0.01,0.1,1,10]

best_depth  = max(depths,  key=lambda d: safe_cv_f1(lambda: MyDecisionTree(max_depth=d)))
best_lambda = max(lambdas, key=lambda l: safe_cv_f1(lambda: MyLogisticRegression(reg=l)))

# --- save ALL results ---
rows = []

for d in depths:
    f1 = safe_cv_f1(lambda: MyDecisionTree(max_depth=d))
    rows.append({"model": "tree", "param": d, "mean_f1": f1})

for l in lambdas:
    f1 = safe_cv_f1(lambda: MyLogisticRegression(reg=l))
    rows.append({"model": "logreg", "param": l, "mean_f1": f1})

pd.DataFrame(rows).to_csv(fig_dir/"cv_table.csv", index=False)

# --- save best params ---
(data_dir/"best_params.json").write_text(
    json.dumps({"max_depth": int(best_depth), "lambda": float(best_lambda)}, indent=2)
)


(data_dir/"best_params.json").write_text(
    json.dumps({"max_depth": int(best_depth), "lambda": float(best_lambda)}, indent=2)
)

print("✓ wrote figures/cv_table.csv   and   data/best_params.json")