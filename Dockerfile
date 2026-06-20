FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates cron && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    fastapi[standard] httpx pyyaml "numpy<2" "huggingface_hub<1.0" \
    pyannote-audio==3.1.1 && \
    rm -rf /root/.cache /tmp/* && \
    sed -i 's/np\.NaN/np.nan/g' /usr/local/lib/python3.11/site-packages/pyannote/audio/core/inference.py && \
    sed -i 's/np\.NaN/np.nan/g' /usr/local/lib/python3.11/site-packages/pyannote/audio/tasks/segmentation/mixins.py && \
    sed -i 's/np\.NaN/np.nan/g' /usr/local/lib/python3.11/site-packages/pyannote/audio/tasks/segmentation/speaker_diarization.py

COPY app/ .
COPY reporting/ /app/reporting/
COPY entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
EXPOSE 8443
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8443"]
