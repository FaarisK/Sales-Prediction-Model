"""
phase4_split_data.py
Splits features.csv → train / val / test (60 / 20 / 20) with stratification.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
df   = pd.read_csv(ROOT / "data" / "features.csv")

X = df.drop(columns=["high_demand"])
y = df["high_demand"]

# 1) train (60%) vs temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=42)

# 2) val (20%) vs test (20%) — still stratified because each class now ≥2
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# save
pd.concat([X_train, y_train], axis=1).to_csv(ROOT/"data"/"train.csv", index=False)
pd.concat([X_val,   y_val],   axis=1).to_csv(ROOT/"data"/"val.csv",   index=False)
pd.concat([X_test,  y_test],  axis=1).to_csv(ROOT/"data"/"test.csv",  index=False)

print("✓  train.csv :", len(X_train), "rows")
print("✓  val.csv   :", len(X_val),   "rows")
print("✓  test.csv  :", len(X_test),  "rows")