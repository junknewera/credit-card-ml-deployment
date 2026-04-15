FROM python:3.12-slim

WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода и моделей
COPY app/ ./app/
COPY models/*.pkl ./models/

EXPOSE 5000

# Запуск приложения
CMD ["python", "-m", "flask", "--app", "app.api", "run", "--host", "0.0.0.0", "--port", "5000"]
