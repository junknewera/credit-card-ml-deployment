import math

from app.model_handler import load_model, predict


def test_model_loads_and_predicts():
    model = load_model("models/model_v1.pkl")
    payload = {
        "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 37,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 46990, "BILL_AMT2": 48233, "BILL_AMT3": 49291,
        "BILL_AMT4": 28314, "BILL_AMT5": 28959, "BILL_AMT6": 29547,
        "PAY_AMT1": 2000, "PAY_AMT2": 2019, "PAY_AMT3": 1200,
        "PAY_AMT4": 1100, "PAY_AMT5": 1069, "PAY_AMT6": 1000,
    }
    result = predict(model, payload)
    assert result["prediction"] in {0, 1}
    assert 0.0 <= result["probability"] <= 1.0
    assert not math.isnan(result["probability"])
