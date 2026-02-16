FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY artifacts /app/artifacts

ENV FAILURE_MODEL=/app/artifacts/pump_failure_lgbm.joblib
ENV TTB_MODEL=/app/artifacts/pump_ttb_lgbm.joblib
ENV SCHEMA_PATH=/app/artifacts/feature_schema.json

# Rust detection model selection (mobilenet or clip)
ENV RUST_MODEL_TYPE=mobilenet
ENV RUST_LABELS=/app/artifacts/rust_labels.json

EXPOSE 8000

# Multi-worker for higher throughput
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
