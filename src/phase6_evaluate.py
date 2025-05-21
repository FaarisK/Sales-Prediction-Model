"""
phase6_evaluate.py  –  final clean version
----------------------------------------------------
• Retrain best‑param models on train+val
• Evaluate on test.csv
• Save confusion‑matrix PNGs + results.json
"""

from pathlib import Path
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

# ── local models ────────────────────────────────────────────────────
from my_decision_tree import MyDecisionTree
from my_logistic_regression import MyLogisticRegression
# (If you prefer `python -m src.phase6_evaluate`, change the two lines above to
#  from src.my_decision_tree import MyDecisionTree, etc.)

# ── paths ───────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── load splits ─────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
val   = pd.read_csv(DATA_DIR / "val.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")
print("Loaded rows  –  train:", len(train), "val:", len(val), "test:", len(test))

trainval = pd.concat([train, val], ignore_index=True)

# ── build feature matrices (drop text col, fill NaNs) ───────────────
drop_cols = ["high_demand", "date"]           # drop label + non‑numeric
X_tv_df   = trainval.drop(columns=drop_cols)
X_test_df = test.drop(columns=drop_cols)

numeric_means = X_tv_df.mean()                # means of numeric cols only
X_tv   = X_tv_df.fillna(numeric_means).to_numpy(dtype=float)
X_test = X_test_df.fillna(numeric_means).to_numpy(dtype=float)

y_tv   = trainval["high_demand"].to_numpy()
y_test = test["high_demand"].to_numpy()

# ── best hyper‑params ───────────────────────────────────────────────
best = json.loads((DATA_DIR / "best_params.json").read_text())
d_depth, lr_lam = best["max_depth"], best["lambda"]
print("Best params  –  tree depth:", d_depth, "| lr lambda:", lr_lam)

# ── train models ────────────────────────────────────────────────────
tree = MyDecisionTree(max_depth=d_depth, min_samples=2, random_state=42).fit(X_tv, y_tv)
lr   = MyLogisticRegression(lr=0.1, n_iter=3000, reg=lr_lam).fit(X_tv, y_tv)
print("Models retrained on train+val")

# ── predict & probabilities ────────────────────────────────────────
pred_tree = tree.predict(X_test)
pred_lr   = lr.predict(X_test)

prob_tree = pred_tree                      # deterministic (0/1); fine for tiny set
prob_lr   = lr.predict_proba(X_test)

# ── metrics helper ─────────────────────────────────────────────────
def metric_dict(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:                     # occurs if only one class in y_prob
        auc = float("nan")
    return dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, zero_division=0),
        recall    = recall_score(y_true, y_pred, zero_division=0),
        f1        = f1_score(y_true, y_pred, zero_division=0),
        roc_auc   = auc
    )

m_tree = metric_dict(y_test, pred_tree, prob_tree)
m_lr   = metric_dict(y_test, pred_lr,   prob_lr)

# ── confusion‑matrix plots ─────────────────────────────────────────
def save_cm(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["0","1"]); ax.set_yticklabels(["0","1"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center")
    ax.set_title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)

save_cm(y_test, pred_tree, "Decision Tree", FIG_DIR/"cm_tree.png")
save_cm(y_test, pred_lr,   "Logistic Regression", FIG_DIR/"cm_logreg.png")

# ── save metrics JSON ──────────────────────────────────────────────
results_path = DATA_DIR / "results.json"
with open(results_path, "w") as f:
    json.dump({"tree":   {"params":{"max_depth":d_depth}, **m_tree},
               "logreg": {"params":{"lambda":lr_lam},     **m_lr}}, f, indent=2)
print("Saved", results_path)

print("\n✓  Evaluation phase complete")
