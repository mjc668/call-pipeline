FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    fastapi[standard] httpx pyyaml \
    pyannote-audio==3.1.1 && \
    pip uninstall -y \
        speechbrain pytorch-lightning lightning optuna \
        pandas matplotlib scikit-learn tensorboardX \
        sqlalchemy alembic pytorch-metric-learning \
        2>/dev/null && \
    rm -rf /root/.cache /tmp/*

COPY app/ .

EXPOSE 8443
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8443"]
