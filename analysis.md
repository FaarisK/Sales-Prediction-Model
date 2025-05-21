## 1  Objective
Predict whether each day is **High Demand** (> 14‑day mean unit sales).

## 2  Data Snapshot
* 14 days: Apr 14 – Apr 27 2025 (Richardson, TX)
* 19 features: lagged unit sales, 7‑day mean, weather, holiday, promo, …
* Split 60 / 20 / 20 → 8 train, 3 val, 3 test

## 3  Model Selection
| Model | Grid | Best Param | CV F1 |
|-------|------|-----------|-------|
| Decision Tree | depth 2–6 | **2** | see cv_table.csv |
| Logistic Reg | λ 0.01–10 | **0.01** | see cv_table.csv |

## 4  Test‑Set Metrics
| Metric | Tree | LogReg |
|--------|-----:|------:|
| Accuracy | **0.67** | 0.67 |
| Precision | **1.00** | 1.00 |
| Recall | 0.50 | 0.50 |
| **F1** | **0.67** | 0.67 |
| ROC‑AUC | 0.75 | 0.75 |

<figure>
  <img src="../figures/cm_tree.png" width="240">
  <img src="../figures/cm_logreg.png" width="240">
</figure>

## 5  Interpretation
* Tree and LogReg tie because the tiny dataset forces the tree to depth 2 ⇒ nearly linear.
* Precision = 1.0 (no false alarms) but Recall = 0.5 ⇒ we miss half the true spikes.

## 6  Limitations & Next Steps
1. Collect ≥ 90 days to stabilise recall.
2. Add dynamic price & competitor traffic.
3. Try Gradient‑Boosted Trees for non‑linear holiday × promo effects.
