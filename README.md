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