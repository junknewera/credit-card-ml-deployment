import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from flask import Flask, g, jsonify, request

from app.model_handler import load_model, predict

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Логирование в JSON
LOG = logging.getLogger("credit_default_service")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
LOG.addHandler(handler)
LOG.setLevel(logging.INFO)

# Загрузка моделей при старте
app = Flask(__name__)
models = {
    "v1": load_model(MODEL_DIR / "model_v1.pkl"),
    "v2": load_model(MODEL_DIR / "model_v2.pkl"),
}


@app.before_request
def _start_timer():
    g.request_id = request.headers.get("X-Request-Id") or str(uuid4())
    g.started_at = time.perf_counter()


@app.after_request
def _log_request(response):
    elapsed = round((time.perf_counter() - g.get("started_at", 0)) * 1000, 2)
    LOG.info(json.dumps({
        "request_id": g.get("request_id"),
        "method": request.method,
        "path": request.path,
        "status": response.status_code,
        "latency_ms": elapsed,
    }))
    return response


@app.get("/health")
def health():
    """Проверка здоровья сервиса"""
    return jsonify({"status": "ok"})


@app.get("/models")
def models_info():
    """Информация о загруженных моделях"""
    info = []
    for version, model in models.items():
        info.append({
            "version": version,
            "estimator": type(model).__name__,
        })
    return jsonify({"models": info})


@app.post("/predict")
def predict_route():
    """Эндпоинт для предсказания дефолта"""
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    version = data.get("model_version", "v1")
    if version not in ("v1", "v2"):
        return jsonify({"error": "model_version must be v1 or v2"}), 400

    return _do_predict(version, data)


@app.post("/predict/v2")
def predict_v2_route():
    """Эндпоинт для предсказания через модель v2"""
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    return _do_predict("v2", data)


def _do_predict(version, data):
    try:
        result = predict(models[version], data)
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "prediction": result["prediction"],
        "probability": round(result["probability"], 6),
        "model_version": version,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
