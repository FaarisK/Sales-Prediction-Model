Walmart High Demand vs Normal Demand 
by: Faaris Khan 
Project Overview

The raw training data (~250 MB CSV) is hosted on Google Drive  
to keep the Git repository lightweight:
https://drive.google.com/drive/folders/1uxjxqGa-SC-W5NN_asjqQOB3LjvT4jmX?usp=drive_link

One or two paragraphs summarizing the goal (binary high-demand prediction), data (14 days of POS + weather + promos), and approach (custom Decision Tree & Logistic Regression).

Repository Structure:
High-level tree of folders/files, e.g.
├── src/                # Python scripts for each phase & model code
├── data/               # CSV splits, params.json, results.json
├── figures/            # Confusion matrices, cv_table.csv
├── report/             # analysis.md (final write-up)
└── README.md           # This file

Setup & Dependencies:
Python version (3.9)
Virtual-env instructions (python3 -m venv .venv && source .venv/bin/activate)
pip install -r requirements.txt (list: pandas, numpy, scikit-learn, matplotlib)

How to Run:
Recreate the virtual environment (can skip, only if .venv is missing):
python3 -m venv .venv

Activate it: 
source .venv/bin/activate

Run: 
pip install -r requirements.txt


Run your script:
Phase 1–6 commands in order:
python src/phase1_build_data.py
python src/phase2_encode_scale.py
python src/phase4_split_data.py
python src/phase5_quick.py
python src/phase6_evaluate.py
Where outputs appear (data/, figures/)

Results Summary:
Best hyperparameters, test metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
Location of plots and JSON
