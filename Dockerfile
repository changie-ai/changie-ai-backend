# Use the same Python as your Mac (3.13)
FROM python:3.13-slim

# Install system-level audio dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    rubberband-cli \
    libsndfile1 \
    sox \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN pip install --no-cache-dir poetry

# Configure Poetry to install into system env
RUN poetry config virtualenvs.create false

# Install Python dependencies
RUN poetry install --no-interaction --no-ansi

# Copy the rest of the app
COPY . /app

# Expose Render port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
