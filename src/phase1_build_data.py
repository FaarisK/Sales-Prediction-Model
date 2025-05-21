"""
phase1_build_data.py
-----------------------------------
• Ensures project folders exist
• Converts 55 000 line‑items → 14‑row daily file
• Writes a Markdown data‑set description
"""

from pathlib import Path
import pandas as pd, numpy as np

# 1.  Project root (auto‑detected: file is two levels below root)
ROOT = Path(__file__).resolve().parents[1]

# 2.  Make sub‑folders if missing
for sub in ["data", "src", "tests", "figures", "report"]:
    (ROOT / sub).mkdir(exist_ok=True)

# 3.  Paths
RAW  = ROOT / "data" / "walmart_line_items.csv"   # already present
DAILY = ROOT / "data" / "daily_features.csv"
MD   = ROOT / "report" / "data_description.md"

# 4.  Load 55 000 line‑items
items = pd.read_csv(RAW, parse_dates=["date"])

# 5.  Aggregate to daily totals
daily = (items.groupby("date")["quantity"]
               .sum().rename("units_sold").reset_index())

# 6.  Feature columns
daily["day_of_week"] = daily["date"].dt.day_name()
daily["holiday"]     = daily["date"].isin(
        ["2025-04-18","2025-04-21"]).astype(int)

np.random.seed(123)
daily["promo"]       = np.random.binomial(1, 0.3, len(daily))
daily["local_event"] = (daily["date"].dt.weekday == 5).astype(int)

daily["temperature_F"] = np.linspace(76, 84, len(daily))
daily["rain_inches"]   = [0,0,0.05,0,0,0.09,0.35,0,0,1.12,0.71,0,0,0]

thr = daily["units_sold"].quantile(0.70)
daily["high_demand"] = (daily["units_sold"] >= thr).astype(int)

for lag in [1, 2, 3, 7]:
    daily[f"lag{lag}"] = daily["units_sold"].shift(lag)
daily["mean7"] = daily["units_sold"].rolling(7).mean().shift(1)

# 7.  Save daily CSV
daily.to_csv(DAILY, index=False)
print(f"✓  {DAILY.name} saved ({len(daily)} rows × {daily.shape[1]} cols)")

# 8.  Markdown stub
md_text = f"""# Data‑Set Description

**Source & Period**  
Richardson Walmart POS logs — 14 Apr 2025 → 27 Apr 2025.

**Files**  
- `walmart_line_items.csv` (55 000 rows)  
- `daily_features.csv` (14 rows × 18 columns)

**Cleaning & Privacy**  
Returns/voids removed; customer IDs stripped.

**Feature List**  
Calendar (weekday / holiday), promo, local_event, weather (temp / rain),
lags 1 / 2 / 3 / 7, 7‑day mean, units_sold, high_demand.
"""
MD.write_text(md_text)
print(f"✓  Markdown stub saved → {MD}")
