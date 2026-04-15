"""Загрузка и использование модели."""

import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def load_model(path):
    return joblib.load(path)


def preprocess_input(data):
    """Преобразование JSON в DataFrame."""
    row = {col: [data[col]] for col in FEATURE_COLUMNS}
    return pd.DataFrame(row)


def predict(model, data):
    features = preprocess_input(data)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    return {"prediction": prediction, "probability": probability}
