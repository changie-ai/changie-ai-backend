FROM python:3.11-slim

WORKDIR /app

# ---- system audio deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    g++ \
    ffmpeg \
    rubberband-cli \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*
# ---------------------------

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements-fly.txt .

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements-fly.txt

COPY . .

CMD ["python", "main.py"]