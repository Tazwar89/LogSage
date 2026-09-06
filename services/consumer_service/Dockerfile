FROM --platform=$BUILDPLATFORM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY libs/logsage_common /app/libs/logsage_common
RUN uv pip install --system --no-cache -e /app/libs/logsage_common

COPY services/consumer_service/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY services/consumer_service/app ./app

CMD ["python", "-m", "app.main"]