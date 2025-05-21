# Data‑Set Description

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
