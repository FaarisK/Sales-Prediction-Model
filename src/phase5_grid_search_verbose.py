"""
phase5_grid_search_verbose.py
---------------------------------------------
• Verifies folders & files exist
• Runs 5‑fold CV grid‑search
• Prints EVERY step
• Saves cv_table.csv  &  best_params.json
Author: Faaris Khan
"""

# ---------- sanity checks ----------
from pathlib import Path, sys

ROOT = Path(__file__).resolve().parents[1]
print("ROOT:", ROOT)

fig_dir  = ROOT / "figures"
data_dir = ROOT / "data"
fig_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)

for f in ["train.csv"]:
    p = data_dir / f
    if not p.exists():
        sys.exit(f"❌  Missing {p}. Run phase4_split_data.py first.")

print("✓  folders & train.csv verified")

# ---------- imports (fail fast) ----------
try:
    import pandas as pd, numpy as np, json
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
except ModuleNotFoundError as e:
    sys.exit(f"❌  Missing package: {e}. Run  'pip install pandas numpy scikit-learn'")

from my_decision_tree import MyDecisionTree     # relative imports (works when run with python src/…)
from my_logistic_regression import MyLogisticRegression

# ---------- load data ----------
train_df = pd.read_csv(data_dir / "train.csv")
X = train_df.drop(columns=["high_demand"]).values
y = train_df["high_demand"].values
print("✓  loaded train.csv   shape:", X.shape)

# tiny dataset guardrail
if len(train_df) < 5:
    sys.exit("❌  Train set too small for 5‑fold CV. Check phase4 split.")

# ---------- parameter grids ----------
tree_depths = [2, 3, 4, 5, 6]
lr_lambdas  = [0.01, 0.1, 1, 10]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []

print("\nRunning Decision‑Tree grid …")
for d in tree_depths:
    fold_scores = []
    for fold,(tr,va) in enumerate(cv.split(X, y), 1):
        model = MyDecisionTree(max_depth=d, min_samples=2, random_state=42)
        model.fit(X[tr], y[tr])
        score = f1_score(y[va], model.predict(X[va]))
        fold_scores.append(score)
        print(f"  depth={d:<2} fold={fold}  f1={score:.3f}")
    rows.append({"model":"tree","param":d,"mean_f1":np.mean(fold_scores)})

print("\nRunning Logistic‑Regression grid …")
for lam in lr_lambdas:
    fold_scores = []
    for fold,(tr,va) in enumerate(cv.split(X, y), 1):
        model = MyLogisticRegression(lr=0.1, n_iter=3000, reg=lam)
        model.fit(X[tr], y[tr])
        score = f1_score(y[va], model.predict(X[va]))
        fold_scores.append(score)
        print(f"  λ={lam:<4} fold={fold}  f1={score:.3f}")
    rows.append({"model":"logreg","param":lam,"mean_f1":np.mean(fold_scores)})

# ---------- save results ----------
import pandas as pd
cv_df = pd.DataFrame(rows)
cv_path = fig_dir / "cv_table.csv"
cv_df.to_csv(cv_path, index=False)
print("\n✓  wrote", cv_path.relative_to(ROOT))

best_tree = cv_df[cv_df.model=="tree"].sort_values("mean_f1", ascending=False).iloc[0]
best_lr   = cv_df[cv_df.model=="logreg"].sort_values("mean_f1", ascending=False).iloc[0]

best = {"max_depth": int(best_tree.param), "lambda": float(best_lr.param)}
(best_path := data_dir / "best_params.json").write_text(json.dumps(best, indent=2))
print("✓  wrote", best_path.relative_to(ROOT))
print("✓  Grid‑search finished   best_depth:", best['max_depth'], " best_lambda:", best['lambda'])
