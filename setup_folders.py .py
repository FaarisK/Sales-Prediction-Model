from pathlib import Path
base = Path("/Users/faaris/Public/ProjectDB/walmart-demand")   # adjust if needed

# create sub‑folders if they don't exist
for sub in ["data", "src", "tests", "figures", "report"]:
    (base / sub).mkdir(parents=True, exist_ok=True)

print("Folders ready:", [p.name for p in base.iterdir()])