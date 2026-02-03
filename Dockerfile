FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    rubberband-cli \
    libsndfile1 \
    sox \
    git \
    build-essential \
    libfftw3-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]