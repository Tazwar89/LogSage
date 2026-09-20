FROM ghcr.io/tazwar89/logsage/base:latest
WORKDIR /app

COPY services/ingestion_service/requirements.txt .
RUN uv pip install --system --no-cache --index-strategy unsafe-best-match -r requirements.txt

COPY services/ingestion_service/app ./app

# Trained sequence-anomaly model (deeplog.pt, drain_sequence.state, meta.json).
# Produce it with scripts/train_sequence_model.py and commit models/sequence/ first.
COPY models/sequence ./models/sequence
ENV SEQUENCE_MODEL_DIR=/app/models/sequence

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]