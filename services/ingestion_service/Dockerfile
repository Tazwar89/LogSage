FROM ghcr.io/tazwar89/logsage/base:latest
WORKDIR /app

COPY services/ingestion_service/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY services/ingestion_service/app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]