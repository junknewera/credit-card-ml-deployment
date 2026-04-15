"""
Скрипт обучения моделей.
Обучает две модели (CatBoost и RandomForest) и сохраняет их в models/.
"""

import pathlib

import joblib
import kagglehub
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    # Загрузка датасета
    path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")
    csv_path = next(pathlib.Path(path).rglob("*.csv"))
    df = pd.read_csv(csv_path)

    # Целевая переменная
    target = "default.payment.next.month"
    if target not in df.columns:
        target = "DEFAULT"

    features = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Модель v1 - CatBoost
    print("Обучение модели v1 (CatBoost)...")
    model_v1 = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=100,
    )
    model_v1.fit(X_train, y_train)
    joblib.dump(model_v1, "models/model_v1.pkl")
    print("Модель v1 сохранена")

    # Модель v2 - RandomForest
    print("Обучение модели v2 (RandomForest)...")
    model_v2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
    )
    model_v2.fit(X_train, y_train)
    joblib.dump(model_v2, "models/model_v2.pkl")
    print("Модель v2 сохранена")

    print("Готово")


if __name__ == "__main__":
    main()
