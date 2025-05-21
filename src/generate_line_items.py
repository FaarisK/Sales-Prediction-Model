import pandas as pd, numpy as np
from datetime import date, timedelta
import random, itertools

START = date(2025, 4, 14)       # 14‑day window
DAYS  = 14
ROWS  = 55_000

SKUS = ["bananas","milk","bread","eggs","apples","chicken breast","ground beef",
        "rice","cereal","cola","yogurt","cheddar cheese","lettuce","tomatoes",
        "onions","potatoes","orange juice","paper towels","toilet paper",
        "coffee","tea","chips","cookies","frozen pizza","ice cream","salsa",
        "pasta","pasta sauce","peanut butter","jelly"]

dates_cycle = itertools.cycle([START + timedelta(d) for d in range(DAYS)])
records = []
for _ in range(ROWS):
    d = next(dates_cycle)
    txn = f"{d:%Y%m%d}_{random.randint(1,1200):04d}"
    item = random.choice(SKUS)
    qty  = random.randint(1,5)
    records.append((d, txn, item, qty))

df = pd.DataFrame(records, columns=["date","transaction_id","item","quantity"])
df.to_csv("data/walmart_line_items.csv", index=False)
print("Saved data/walmart_line_items.csv with", len(df), "rows")
