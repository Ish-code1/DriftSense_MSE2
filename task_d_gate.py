# Usage: python task_d_gate.py 3
import json
from datetime import datetime

import joblib

from common import *

MIN_GAIN = 0.01         # candidate must beat champion by 1% macro-F1 on new holdout
MAX_REGRESSION = 0.02   # and lose at most 2% accuracy on the old test set

X, y = load_batch(BATCH)
_, X_ho, _, y_ho = new_split(X, y)
_, X_old, _, y_old = old_split()

champion = joblib.load("models/driftsense.joblib")
candidate = joblib.load("models/candidate.joblib")

champ_f1 = scores(champion, X_ho, y_ho)["f1"]
cand_f1 = scores(candidate, X_ho, y_ho)["f1"]

eligible = (cand_f1 - champ_f1) >= MIN_GAIN and drop <= MAX_REGRESSION

decision = {
    "time": datetime.now().isoformat(),
    "metric": "macro_f1",
    "champion_f1": champ_f1,
    "candidate_f1": cand_f1,
    "old_test_accuracy_drop": drop,
    "decision": "PROMOTE_ELIGIBLE" if eligible else "RETAIN_CHAMPION",
}

with open("reports/gate_decision.json", "w") as f:
    json.dump(decision, f, indent=2)
with open("models/decision_log.jsonl", "a") as f:   # audit trail
    f.write(json.dumps(decision) + "\n")
print(decision)