FROM python:3.11-slim

WORKDIR /app

# ---- system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    g++ \
    ffmpeg \
    rubberband-cli \
    libsndfile1 \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*
# --------------------

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# install poetry
RUN pip install --no-cache-dir poetry

# copy lockfiles FIRST (important for Docker caching)
COPY pyproject.toml poetry.lock* /app/

# install deps EXACTLY as locked
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# now copy the rest of the app
COPY . /app

CMD ["python", "main.py"]
