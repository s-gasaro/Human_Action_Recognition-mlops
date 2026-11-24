FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 PORT=8000 \
    DB_PATH=/app/shared/app.db UPLOAD_DIR=/app/shared/uploads
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY models ./models
COPY data/eval ./data/eval
COPY data/replay ./data/replay
COPY data/splits.csv ./data/splits.csv
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
