FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY libs/logsage_common /app/libs/logsage_common
RUN uv pip install --system --no-cache -e /app/libs/logsage_common

COPY services/analysis-service/requirements.txt .
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY services/analysis-service/app ./app
COPY services/analysis-service/data ./data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]