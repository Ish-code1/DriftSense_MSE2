import json, joblib
from datetime import datetime
from common import *

MIN_GAIN = 0.01         
MAX_REGRESSION = 0.02   

X, y = load_batch(BATCH)
_, X_ho, _, y_ho = new_split(X, y)
_, X_old, _, y_old = old_split()

champion = joblib.load("models/driftsense.joblib")
candidate = joblib.load("models/candidate.joblib")

champ_f1 = scores(champion, X_ho, y_ho)["f1"]
cand_f1 = scores(candidate, X_ho, y_ho)["f1"]
drop = scores(champion, X_old, y_old)["accuracy"] - scores(candidate, X_old, y_old)["accuracy"]

eligible = (cand_f1 - champ_f1) >= MIN_GAIN and drop <= MAX_REGRESSION

decision = {
    "time": datetime.now().isoformat(),
    "metric": "macro_f1",
    "champion_f1": champ_f1,
    "candidate_f1": cand_f1,
    "old_test_accuracy_drop": drop,
    "decision": "PROMOTE_ELIGIBLE" if eligible else "RETAIN_CHAMPION",
}

json.dump(decision, open("reports/gate_decision.json", "w"), indent=2)
open("models/decision_log.jsonl", "a").write(json.dumps(decision) + "\n")   # audit trail
print(decision)
