FROM ghcr.io/tazwar89/logsage/base:latest
WORKDIR /app

COPY services/analysis_service/requirements.txt .
# The base image sets UV_EXTRA_INDEX_URL to the PyTorch CPU index, which carries
# outdated requests/urllib3. Without unsafe-best-match, uv only considers that
# index for those packages and the resolve fails. torch is already installed in
# the base, so the extra index is not needed for anything here.
RUN uv pip install --system --no-cache --index-strategy unsafe-best-match -r requirements.txt

COPY services/analysis_service/app ./app
COPY services/analysis_service/data ./data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]