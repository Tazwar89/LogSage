FROM logsage-base:latest
WORKDIR /app

COPY services/analysis_service/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY services/analysis_service/app ./app
COPY services/analysis_service/data ./data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]