# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — ShopNow AI Voice Agent
# Build : docker build -t shopnow-agent .
# Run   : docker run -p 8000:8000 --env-file .env shopnow-agent
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# --- system deps -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- working directory -----------------------------------------------------
WORKDIR /app

# --- install Python dependencies first (layer-cached separately) -----------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- copy application source -----------------------------------------------
COPY . .

# --- runtime configuration -------------------------------------------------
# The real .env (with OPENAI_API_KEY etc.) should be passed at runtime via
# --env-file or individual -e flags.  We do NOT bake secrets into the image.
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV LOG_LEVEL=info

EXPOSE 8000

# --- health-check -----------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# --- start server -----------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
