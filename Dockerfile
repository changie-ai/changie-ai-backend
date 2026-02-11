FROM python:3.11-slim

WORKDIR /app

# ---- system deps for audio ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    g++ \
    ffmpeg \
    sox \
    rubberband-cli \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for caching
COPY requirements-fly.txt .

# Install Python deps
RUN pip install --no-cache-dir --retries 10 --timeout 60 -r requirements-fly.txt

# Copy the app
COPY . .

# Run the app
CMD ["python", "main.py"]