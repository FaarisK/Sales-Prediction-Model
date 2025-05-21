# src/phase2_encode_scale.py
# --------------------------------------------------------
# Phase 2: encode `day_of_week` and scale numeric columns
# Outputs:
#   • data/features.csv   (18 columns)
#   • data/scaler.pkl     (StandardScaler object)
# --------------------------------------------------------

from pathlib import Path
import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

ROOT  = Path(__file__).resolve().parents[1]          # project root
DAILY = ROOT / "data" / "daily_features.csv"
OUT_CSV = ROOT / "data" / "features.csv"
OUT_PKL = ROOT / "data" / "scaler.pkl"

# ------------------------------------------------------------------
# 1.  Load the 14‑row daily file
# ------------------------------------------------------------------
df = pd.read_csv(DAILY)

# ------------------------------------------------------------------
# 2.  One‑hot‑encode day_of_week  (drop first to avoid dummy trap)
#     New scikit‑learn (>=1.2) expects sparse_output instead of sparse
# ------------------------------------------------------------------
ohe = OneHotEncoder(drop="first", sparse_output=False)
dow_dummies = ohe.fit_transform(df[["day_of_week"]])
dow_cols = ohe.get_feature_names_out(["day_of_week"])   # e.g., day_of_week_Tuesday …

df_enc = pd.concat(
    [df.drop(columns=["day_of_week"]).reset_index(drop=True),
     pd.DataFrame(dow_dummies, columns=dow_cols)], axis=1
)

# ------------------------------------------------------------------
# 3.  Standard‑scale numeric columns (leave binaries & target untouched)
# ------------------------------------------------------------------
binary_cols = ["holiday", "promo", "local_event", "high_demand"]
num_cols = [c for c in df_enc.select_dtypes(include="number").columns
            if c not in binary_cols]

scaler = StandardScaler().fit(df_enc[num_cols])
df_enc[num_cols] = scaler.transform(df_enc[num_cols])

# ------------------------------------------------------------------
# 4.  Save outputs
# ------------------------------------------------------------------
df_enc.to_csv(OUT_CSV, index=False)
joblib.dump(scaler, OUT_PKL)

print(f"✓  {OUT_CSV.name} saved  →  rows: {len(df_enc)}, cols: {df_enc.shape[1]}")
print(f"✓  Scaler object saved  →  {OUT_PKL.name}")
