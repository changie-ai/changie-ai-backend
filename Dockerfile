FROM python:3.11-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=120

# ---- System deps for audio DSP (harmonizer / pitch) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    ffmpeg \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*
# --------------------------------------------------------

COPY requirements-fly.txt /app/requirements-fly.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install \
    --retries 20 \
    --timeout 180 \
    --prefer-binary \
    --no-build-isolation \
    -r requirements-fly.txt

COPY . /app

CMD ["python", "main.py"]