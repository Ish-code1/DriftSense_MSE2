# DriftSense

ML-powered gas sensor monitoring. Classifies gas samples and detects sensor drift in incoming data, deployed as a containerized API on AWS.

Gas sensors degrade over time, so readings shift and a classifier trained on old data gets less accurate. DriftSense serves a gas classifier and flags when new data no longer matches the training distribution.

## Dataset
UCI Gas Sensor Array Drift Dataset: 13,910 samples, 16 sensors x 8 features = 128 features, 6 gases, collected over 36 months in 10 batches. Batch 1 is the reference period. Later batches show real sensor drift.

## Pipeline
MLflow -> FastAPI -> Pytest -> Docker -> GitHub Actions (OIDC) -> Amazon ECR -> EC2

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service status |
| `/health` | GET | Health check + model version |
| `/predict` | POST | Classify one sample (128 features) |
| `/drift` | POST | Check a batch of samples (min 10) for drift vs training data |

**Predict**
```json
POST /predict
{ "features": [ ...128 floats... ] }

-> { "prediction": 2, "gas": "acetaldehyde" }
```

**Drift**
```json
POST /drift
{ "samples": [ [...128 floats...], ... ] }

-> { "drift_detected": true, "drifted_feature_share": 0.62 }
```
Drift check: per-feature batch mean vs training mean, measured in training std. If more than 30% of features shift by more than 1 std, drift is flagged.

## Training
Trained on batch 1 (Random Forest). The same model is then evaluated on batches 2-10 to measure accuracy decay from drift. Everything is tracked in MLflow: params, metrics, model registry, and a quality gate (min accuracy 0.9).

![Accuracy per batch](docs/accuracy_per_batch.png)

Results:
- Batch 1 test accuracy: **XX%**
- Accuracy on batch 10: **XX%**

## Run locally
```bash
python -m venv myvenv
source myvenv/bin/activate        # Windows: myvenv\Scripts\activate
pip install -r requirements.txt
pytest
uvicorn main:app --port 8000
```

## Docker
```bash
docker build -t driftsense .
docker run -p 8000:8000 driftsense
```

## Deployment
Push to `main` triggers GitHub Actions: tests -> Docker build -> push to ECR (OIDC, no stored AWS keys) -> pull and run on EC2. The model is bundled into the image.

## Retraining
Retrain on newer batches and compare the candidate against the current model on a later holdout batch. Promote only if it performs better.

## Tech
Python, scikit-learn, MLflow, FastAPI, Pydantic, Pytest, Docker, GitHub Actions, AWS (ECR, EC2, IAM OIDC)


---

## 📋 MLOps Practical Assessment: Model Quality Monitoring & Retraining Pipeline

This section documents the practical implementation of the model monitoring, evaluation, retraining, and automated quality gate workflow built for DriftSense using incoming batches from the UCI Gas Sensor Array Drift Dataset.

---

### Task A — Prepare New Data
* **Implementation File**: [`task_a_validate.py`](task_a_validate.py) (supported by [`common.py`](common.py))
* **Execution Command**:
  ```bash
  python task_a_validate.py 3
  ```
* **Pipeline Checks & Verification**:
  * **Incoming Stream**: Ingested Batch 3 representing time-shifted sensor operational data.
  * **Schema Validation**: Ensured every sample has exactly 128 continuous numerical features ($16 \text{ sensors} \times 8 \text{ features}$).
  * **Missing Values**: Checked for missing/NaN cells (`missing_cells: 0`).
  * **Duplicate Detection**: Verified zero duplicate sample rows (`duplicate_rows: 0`).
  * **Dataset Size**: **1,586 rows** $\times$ **128 features**.
  * **Target Distribution**: Multi-class balance across the gas classes:
    * Class 0 (Ethanol): 365 samples (23.0%)
    * Class 1 (Ethylene): 490 samples (30.9%)
    * Class 2 (Ammonia): 216 samples (13.6%)
    * Class 3 (Acetaldehyde): 240 samples (15.1%)
    * Class 4 (Acetone): 275 samples (17.3%)
* **Artifact Generated**: Saved to [`reports/data_validation.json`](reports/data_validation.json).

---

### Task B — Evaluate the Existing Model
* **Implementation File**: [`task_b_evaluate.py`](task_b_evaluate.py)
* **Execution Command**:
  ```bash
  python task_b_evaluate.py 3
  ```
* **Evaluation Workflow**:
  * Loaded current champion model artifact from `models/driftsense.joblib` (originally trained on Batch 1 reference data).
  * Generated predictions on the newly verified labelled Batch 3 to evaluate accuracy decay caused by physical sensor drift.
* **Evaluation Metrics on Batch 3**:
  * **Accuracy**: **70.62%** (decayed from 95.51% baseline on Batch 1)
  * **Macro Precision**: **54.00%**
  * **Macro Recall**: **53.83%**
  * **Macro F1-Score**: **50.38%**
* **Artifact Generated**: Saved to [`reports/evaluation_report.json`](reports/evaluation_report.json).

---

### Task C — Retrain the Model
* **Implementation File**: [`task_c_retrain.py`](task_c_retrain.py)
* **Execution Command**:
  ```bash
  python task_c_retrain.py 3
  ```
* **Retraining Pipeline**:
  * **Data Consolidation**: Merged historical baseline data (Batch 1 80% train split) with 70% of the new labelled batch (`X_old_tr` + `X_new_tr`). The remaining 30% of Batch 3 was preserved as a strict holdout set for the quality gate.
  * **Candidate Training**: Fitted a new `RandomForestClassifier(n_estimators=100, random_state=42)` across the combined data matrix.
  * **Artifact Isolation**: Candidate model is saved separately to [`models/candidate.joblib`](models/candidate.joblib), guaranteeing the existing champion (`models/driftsense.joblib`) remains completely untouched during training.
  * **MLflow Tracking**: Logged run in local MLflow experiment `driftsense-retraining`:
    * Run Name: `candidate_batch3`
    * Parameters: `n_estimators: 100`, `train_batches: "1+3"`, `train_rows: 1475`
    * Metrics: Logged holdout accuracy, precision, recall, and F1-score.
    * Model Artifact: Candidate logged using `mlflow.sklearn.log_model`.

---

### Task D — Implement a Quality Gate
* **Implementation File**: [`task_d_gate.py`](task_d_gate.py)
* **Execution Command**:
  ```bash
  python task_d_gate.py 3
  ```
* **Decision Rules & Thresholds**:
  1. **New Holdout Performance**: Candidate must beat Champion by at least 1% Macro-F1 on the new 30% holdout set:
     $$\text{Gain} = \text{F1}_{\text{candidate}} - \text{F1}_{\text{champion}} \ge 0.01$$
  2. **Catastrophic Forgetting Safeguard**: Candidate's accuracy drop on the original Batch 1 test set must not exceed 2%:
     $$\text{Drop} = \text{Acc}_{\text{old, champ}} - \text{Acc}_{\text{old, cand}} \le 0.02$$
* **Selected Metric & Justification**:
  * **Selected Metric**: **Macro F1-Score**.
  * **Justification**: The dataset exhibits multi-class imbalance across 6 gas categories (including toxic gases like ammonia and acetaldehyde). Standard accuracy can be misleading if a model only predicts the majority class (e.g. ethylene). Macro F1 gives equal weight to all gas categories, ensuring that performance improvements reflect genuine classification capability across all gases without masking safety-critical misclassifications.
* **Gate Execution Results**:
  * **Champion F1 (Batch 3 Holdout)**: **0.5002** (50.02%)
  * **Candidate F1 (Batch 3 Holdout)**: **0.8300** (83.00%) $\rightarrow$ **+32.98% improvement!**
  * **Old Test Accuracy Drop**: **-0.0112** (-1.12% drop, i.e., candidate retained strong performance on historical data)
  * **Final Decision**: **`PROMOTE_ELIGIBLE`**
* **Audit Trail**:
  * Stored gate summary in [`reports/gate_decision.json`](reports/gate_decision.json).
  * Appended full decision record with timestamps to [`models/decision_log.jsonl`](models/decision_log.jsonl) for immutable compliance auditing.

---

### Task E — Demonstration and Explanation
* **Complete Workflow Execution Order**:
  ```bash
  # Step 1: Validate incoming data quality
  python task_a_validate.py 3

  # Step 2: Evaluate current champion performance
  python task_b_evaluate.py 3

  # Step 3: Retrain candidate on consolidated data and log to MLflow
  python task_c_retrain.py 3

  # Step 4: Execute quality gate comparison and log promotion decision
  python task_d_gate.py 3
  ```
* **Evidence & Artifacts Summary**:
  * 📄 **Data Validation Report**: `reports/data_validation.json`
  * 📄 **Model Evaluation Report**: `reports/evaluation_report.json`
  * 📦 **Candidate Model Artifact**: `models/candidate.joblib`
  * ⚖️ **Quality Gate Decision**: `reports/gate_decision.json`
  * 📝 **Audit Trail**: `models/decision_log.jsonl`
  * 📊 **MLflow Run Logs**: `./mlruns` (Experiment: `driftsense-retraining`)
