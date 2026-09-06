FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir uv

# Install shared library first (changes less often than service code)
COPY libs/logsage_common /app/libs/logsage_common
RUN uv pip install --system --no-cache -e /app/libs/logsage_common

COPY services/ingestion_service/requirements.txt .
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY services/ingestion_service/app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]