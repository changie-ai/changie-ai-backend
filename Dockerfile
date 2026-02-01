# Use Python 3.13 like your local Mac
FROM python:3.13-slim

# Install system-level audio dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    rubberband-cli \
    libsndfile1 \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock* /app/

# Install poetry
RUN pip install --no-cache-dir poetry

# Configure poetry to install into system env
RUN poetry config virtualenvs.create false

# Install Python dependencies ONLY (no project root)
RUN poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the app
COPY . /app

# Expose Render port
EXPOSE 8080

# Start FastAPI
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

