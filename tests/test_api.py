import numpy as np
from fastapi.testclient import TestClient

from main import N_FEATURES, create_app


class DummyModel:
    def predict(self, X):
        return np.array([2] * len(X))  # always class 2 -> acetaldehyde


def dummy_model_loader():
    return DummyModel()


def dummy_stats_loader():
    return {"mean": [0.0] * N_FEATURES, "std": [1.0] * N_FEATURES}


def make_client(stats_loader=dummy_stats_loader):
    app = create_app(model_loader=dummy_model_loader, stats_loader=stats_loader)
    return TestClient(app)


def test_ready():
    with make_client() as client:
        r = client.get("/")
        assert r.status_code == 200
        assert r.json() == {"status": "Ready to serve"}


def test_health():
    with make_client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "health"


def test_predict_valid():
    with make_client() as client:
        r = client.post("/predict", json={"features": [0.5] * N_FEATURES})
        assert r.status_code == 200
        assert r.json() == {"prediction": 2, "gas": "ammonia"}


def test_predict_wrong_feature_count():
    with make_client() as client:
        r = client.post("/predict", json={"features": [0.5] * 10})
        assert r.status_code == 422


def test_predict_missing_body():
    with make_client() as client:
        r = client.post("/predict", json={})
        assert r.status_code == 422


def test_drift_not_detected():
    with make_client() as client:
        samples = [[0.0] * N_FEATURES for _ in range(10)]
        r = client.post("/drift", json={"samples": samples})
        assert r.status_code == 200
        assert r.json()["drift_detected"] is False


def test_drift_detected():
    with make_client() as client:
        samples = [[5.0] * N_FEATURES for _ in range(10)]
        r = client.post("/drift", json={"samples": samples})
        assert r.status_code == 200
        assert r.json()["drift_detected"] is True
        assert r.json()["drifted_feature_share"] == 1.0


def test_drift_too_few_samples():
    with make_client() as client:
        samples = [[0.0] * N_FEATURES for _ in range(3)]
        r = client.post("/drift", json={"samples": samples})
        assert r.status_code == 422


def test_drift_wrong_feature_count():
    with make_client() as client:
        samples = [[0.0] * 5 for _ in range(10)]
        r = client.post("/drift", json={"samples": samples})
        assert r.status_code == 422


def test_drift_without_reference_stats():
    with make_client(stats_loader=lambda: None) as client:
        samples = [[0.0] * N_FEATURES for _ in range(10)]
        r = client.post("/drift", json={"samples": samples})
        assert r.status_code == 503