# Usage: python task_b_evaluate.py 3
import json, joblib
from common import *

X, y = load_batch(BATCH)
champion = joblib.load("models/driftsense.joblib")

report = {"batch": BATCH, "champion_metrics": scores(champion, X, y)}

json.dump(report, open("reports/evaluation_report.json", "w"), indent=2)
print(report)
