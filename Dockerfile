# syntax=docker/dockerfile:1.7
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    POETA_HOME=/workspace

# Install system dependencies needed for compiling wheels and optional extras like pyserini
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        pkg-config \
        libgl1 \
        libglib2.0-0 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${POETA_HOME}

COPY . ${POETA_HOME}

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e . && \
    python -m compileall .

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
