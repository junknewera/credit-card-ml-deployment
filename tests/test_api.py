from app.api import app


def _payload():
    return {
        "LIMIT_BAL": 90000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 34,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 29239, "BILL_AMT2": 14027, "BILL_AMT3": 13559,
        "BILL_AMT4": 14331, "BILL_AMT5": 14948, "BILL_AMT6": 15549,
        "PAY_AMT1": 1518, "PAY_AMT2": 1500, "PAY_AMT3": 1000,
        "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 5000,
    }


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_models_endpoint():
    client = app.test_client()
    response = client.get("/models")
    assert response.status_code == 200
    body = response.get_json()
    assert "models" in body
    assert {item["version"] for item in body["models"]} == {"v1", "v2"}


def test_predict_default_model():
    client = app.test_client()
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_version"] == "v1"
    assert body["prediction"] in {0, 1}
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_v2_route():
    client = app.test_client()
    response = client.post("/predict/v2", json=_payload())
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_version"] == "v2"


def test_invalid_payload_returns_400():
    client = app.test_client()
    response = client.post("/predict", json={"not": "a valid payload"})
    assert response.status_code == 400
    assert "error" in response.get_json()
