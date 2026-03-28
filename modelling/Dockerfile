# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install a specific pre-built CmdStan version to avoid recompiling from source
# 2.35.0 is stable and avoids the 500MB+ compile overhead of pulling latest
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.35.0', overwrite=False)"

# Strip test files and __pycache__ from site-packages to save ~100MB
RUN find /usr/local/lib/python3.11/site-packages \
    -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages \
    -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages \
    -name "*.pyc" -delete 2>/dev/null || true

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cmdstan /root/.cmdstan

WORKDIR /app

COPY . /app/modelling/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    CMDSTAN=/root/.cmdstan/cmdstan-2.35.0

# Default: XGBoost training. Override CMD to run any script:
#   docker run <image>                                          → XGBoost
#   docker run <image> modelling/models/prophet_model.py        → Prophet
#   docker run <image> modelling/models/optuna_tuning.py ...    → Tuning
#   docker run <image> modelling/scripts/bias_detection.py ...  → Bias check
ENTRYPOINT ["python"]
CMD ["modelling/models/xgboost_model.py"]