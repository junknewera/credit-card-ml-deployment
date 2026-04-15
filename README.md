# Сервис прогнозирования дефолта по кредитным картам

Веб-сервис на Flask, который предсказывает дефолт клиентов по кредитным картам. Датасет - [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

Обучены две модели для A/B-тестирования:

- v1: CatBoostClassifier
- v2: RandomForestClassifier

## Структура репозитория

```
app/                  - Flask-приложение
  api.py              - эндпоинты API
  model_handler.py    - загрузка и использование модели
models/               - сохраненные модели (.pkl)
  train_model.py      - скрипт обучения моделей
tests/                - тесты API и модели
notebooks/            - Jupyter-ноутбук с EDA и обучением
docs/                 - документация (архитектура, A/B-тест)
```

## Запуск локально

Создать виртуальное окружение и установить зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Обучение моделей (если нужно переобучить):

```bash
python models/train_model.py
```

Запуск сервиса:

```bash
python -m flask --app app.api run --host 0.0.0.0 --port 5000
```

## Docker

Сборка образа:

```bash
docker build -t credit-default-service .
```

Запуск контейнера:

```bash
docker run --rm -p 5000:5000 credit-default-service
```

Docker Compose (сервис + nginx):

```bash
docker compose up --build
```

Образ на Docker Hub:

```bash
docker pull junknewera/credit-default-service:latest
```

## API

### GET /health

Проверка что сервис работает.

```bash
curl http://localhost:5000/health
```

Ответ:

```json
{"status": "ok"}
```

### GET /models

Какие модели загружены.

```bash
curl http://localhost:5000/models
```

Ответ:

```json
{"models": [{"version": "v1", "estimator": "CatBoostClassifier"}, {"version": "v2", "estimator": "RandomForestClassifier"}]}
```

### POST /predict

Предсказание дефолта. По умолчанию v1, можно передать `model_version` для выбора.

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 32,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 13000,
    "BILL_AMT4": 12000, "BILL_AMT5": 11000, "BILL_AMT6": 10000,
    "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
    "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000
  }'
```

Ответ:

```json
{"prediction": 0, "probability": 0.051234, "model_version": "v1"}
```

Запрос с моделью v2:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_version": "v2", "LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 32, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 13000, "BILL_AMT4": 12000, "BILL_AMT5": 11000, "BILL_AMT6": 10000, "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000, "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000}'
```

### POST /predict/v2

Отдельный эндпоинт, всегда использует v2. Формат запроса тот же.

### Признаки

| Признак | Описание |
|---------|----------|
| LIMIT_BAL | Кредитный лимит |
| SEX | Пол (1 = мужской, 2 = женский) |
| EDUCATION | Образование (1 = аспирантура, 2 = университет, 3 = школа, 4 = другое) |
| MARRIAGE | Семейное положение (1 = женат/замужем, 2 = не женат, 3 = другое) |
| AGE | Возраст |
| PAY_0 - PAY_6 | Статус платежей за 6 месяцев (-1 = вовремя, 1 = задержка 1 мес, ...) |
| BILL_AMT1 - BILL_AMT6 | Сумма счета за 6 месяцев |
| PAY_AMT1 - PAY_AMT6 | Сумма платежа за 6 месяцев |

## Тесты

```bash
pytest tests/
```

## Документация

- [Архитектура сервиса](docs/ARCHITECTURE.md)
- [План A/B-тестирования](docs/AB_TEST_PLAN.md)

## ONNX-ML

Модели можно конвертировать в ONNX для ускорения инференса. Для sklearn есть `skl2onnx`, у CatBoost встроенный экспорт. После конвертации нужно сверить выходы с оригиналом и дальше использовать ONNX Runtime. Имеет смысл, когда нужен портативный формат или сервинг не на Python.

## uWSGI + NGINX

Flask сам по себе не годится для production - однопоточный dev-сервер. Обычная схема: uWSGI (или Gunicorn) запускает Flask в нескольких воркерах, а NGINX стоит перед ними как обратный прокси, терминирует TLS, раздает статику, балансирует нагрузку. В docker-compose.yml проекта nginx стоит перед сервисом как пример такой связки.
