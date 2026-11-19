from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import numpy as np
import json
import os
from contextlib import asynccontextmanager
from typing import Callable


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:7001"
)
# MLFlow Configuration
# MODEL_URI = "models:/driftsense-classifier@champion"  because after aws deployment, the model uri is different.
MODEL_URI = "model"
# Reference stats (training mean/std) exported along with the model
STATS_PATH = "model/reference_stats.json"

N_FEATURES = 128  # 16 sensors x 8 features
GASES = ["ethanol", "ethylene", "ammonia", "acetaldehyde", "acetone", "toluene"]


def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


def load_reference_stats():
    if not os.path.exists(STATS_PATH):
        return None
    with open(STATS_PATH) as f:
        return json.load(f)


class SensorInput(BaseModel):
    features: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)


class DriftInput(BaseModel):
    # a batch of recent sensor readings
    samples: list[list[float]] = Field(min_length=10)


def create_app(
    model_loader: Callable = load_mlflow_model,
    version_loader: Callable | None = None,
    stats_loader: Callable = load_reference_stats,
):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
        except Exception as ex:
            raise RuntimeError(f"Failed to load model: {ex}") from ex

        app.state.ref_stats = stats_loader()

        if version_loader is None:
            app.state.model_version = os.getenv("MODEL_VERSION", "bundled")
        else:
            app.state.model_version = version_loader()

        yield

    app = FastAPI(
        title="DriftSense API",
        description="Gas sensor classification and drift detection powered by MLFlow",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    def ready_to_serve():
        return {"status": "Ready to serve"}

    @app.get("/health")
    def health_check():
        return {"status": "health"}

    # Endpoint for classifying one gas sensor sample
    @app.post("/predict")
    def predict(data: SensorInput, request: Request):
        features = [data.features]
        prediction = int(request.app.state.model.predict(features)[0])
        return {
            "prediction": prediction,
            "gas": GASES[prediction],
        }

    # Endpoint for checking a batch of recent readings against training data
    @app.post("/drift")
    def drift(data: DriftInput, request: Request):
        stats = request.app.state.ref_stats
        if stats is None:
            raise HTTPException(status_code=503, detail="Reference stats not available")
        batch = np.array(data.samples)
        if batch.ndim != 2 or batch.shape[1] != N_FEATURES:
            raise HTTPException(
                status_code=422,
                detail=f"Each sample needs {N_FEATURES} features",
            )
        ref_mean = np.array(stats["mean"])
        ref_std = np.array(stats["std"]) + 1e-9
        # how far each feature's batch mean moved, in training std units
        z = np.abs(batch.mean(axis=0) - ref_mean) / ref_std
        # share of features that shifted by more than 1 std
        drifted_share = float((z > 1).mean())
        return {
            "drift_detected": drifted_share > 0.3,
            "drifted_feature_share": round(drifted_share, 3),
        }

    return app


app = create_app()
