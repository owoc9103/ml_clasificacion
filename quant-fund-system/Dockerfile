# ---------- Builder ----------
    FROM python:3.11-slim AS builder

    ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
        PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    
    WORKDIR /app
    
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl \
        && rm -rf /var/lib/apt/lists/*
    
    COPY requirements.txt ./
    RUN pip install --upgrade pip && \
        pip wheel --wheel-dir=/wheels -r requirements.txt
    
    # ---------- Runtime ----------
    FROM python:3.11-slim AS runtime
    
    ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
        PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
        PYTHONPATH=/app/src
    
    WORKDIR /app
    
    # xgboost runtime bağımlılığı
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*
    
    # (Opsiyonel) matplotlib ile daha iyi font için:
    # RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu && rm -rf /var/lib/apt/lists/*
    
    COPY --from=builder /wheels /wheels
    RUN pip install --no-cache-dir /wheels/*
    
    # Uygulama dosyaları
    COPY . .
    
    # Sağlık kontrolü
    HEALTHCHECK --interval=1h --timeout=10s --start-period=5s --retries=3 \
      CMD python scripts/health_check.py || exit 1
    
    # Varsayılan komut (paper trading)
    CMD ["python", "scripts/run_live.py", "--mode", "paper"]
    