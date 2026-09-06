FROM --platform=$BUILDPLATFORM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Global environment variables to force CPU-only downloads for all steps
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

COPY libs/logsage_common /app/libs/logsage_common
RUN uv pip install --system --no-cache -e /app/libs/logsage_common

COPY services/ingestion_service/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY services/ingestion_service/app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]