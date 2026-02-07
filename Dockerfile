FROM python:3.11-slim

WORKDIR /app

# ---- system deps + TLS certs ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    g++ \
    ffmpeg \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*
# --------------------------------

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_TRUSTED_HOST=pypi.org

COPY requirements-fly.txt /app/requirements-fly.txt

RUN pip install \
    --retries 40 \
    --timeout 300 \
    --index-url https://pypi.org/simple \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --prefer-binary \
    -r requirements-fly.txt

COPY . /app

CMD ["python", "main.py"]