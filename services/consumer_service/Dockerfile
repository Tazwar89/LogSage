# Does NOT extend logsage-base:latest -- that image bundles logsage_common's
# [vector] extra (torch, sentence-transformers, ~700MB+), which this service
# never uses. Installing logsage_common without the extra here keeps this
# image genuinely lightweight, matching its role as the leanest of the
# three services.
FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY libs/logsage_common /app/libs/logsage_common
RUN uv pip install --system --no-cache -e /app/libs/logsage_common

COPY services/consumer_service/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY services/consumer_service/app ./app

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "app.main"]