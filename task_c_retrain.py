
import joblib, numpy as np
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from common import *

X, y = load_batch(BATCH)
X_new_tr, X_new_ho, y_new_tr, y_new_ho = new_split(X, y)
X_old_tr, _, y_old_tr, _ = old_split()


X_all = np.vstack([X_old_tr, X_new_tr])
y_all = np.concatenate([y_old_tr, y_new_tr])

candidate = RandomForestClassifier(n_estimators=100, random_state=42)
candidate.fit(X_all, y_all)
joblib.dump(candidate, "models/candidate.joblib")     


mlflow.set_experiment("driftsense-retraining")
with mlflow.start_run(run_name=f"candidate_batch{BATCH}"):
    mlflow.log_params({"n_estimators": 100, "train_batches": f"1+{BATCH}", "train_rows": len(X_all)})
    mlflow.log_metrics({f"holdout_{k}": v for k, v in scores(candidate, X_new_ho, y_new_ho).items()})
    mlflow.sklearn.log_model(candidate, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
