# Fraud scoring API image.
#   build:  docker build -t fraud-api:latest .
#   run:    docker run -p 8000:8000 fraud-api:latest      (docs at http://127.0.0.1:8000/docs)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs the OpenMP runtime, which slim images do not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached and only rebuilt when requirements change.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Code and the three model artifacts the API loads (no data, notebooks or MLflow files).
COPY src/ src/
COPY api/ api/
COPY models/final_model.txt models/metadata.json models/policy.json models/

# Run as an unprivileged user; it can only write to its own log folder.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown appuser /app/logs
USER appuser

ENV PYTHONPATH=/app/src:/app \
    FRAUD_MODEL_DIR=/app/models \
    FRAUD_PREDICTION_LOG=/app/logs/predictions.jsonl

EXPOSE 8000

# Docker marks the container unhealthy if the model is not serving.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
