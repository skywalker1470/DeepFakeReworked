from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "request_count" in body
    assert "avg_latency_ms" in body


def test_predict_rejects_missing_file():
    response = client.post("/predict")
    assert response.status_code == 422
