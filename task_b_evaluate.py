# Usage: python task_b_evaluate.py 3
import json

import joblib

from common import *

X, y = load_batch(BATCH)
champion = joblib.load("models/driftsense.joblib")

report = {"batch": BATCH, "champion_metrics": scores(champion, X, y)}

with open("reports/evaluation_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(report)