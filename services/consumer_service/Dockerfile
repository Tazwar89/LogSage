FROM logsage-base:latest
WORKDIR /app

COPY services/consumer_service/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY services/consumer_service/app ./app

CMD ["python", "-m", "app.main"]