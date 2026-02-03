# 1️⃣ Use the same Python version as your Mac
FROM python:3.13-slim

# 2️⃣ Install system audio tools and build dependencies
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

# 3️⃣ Set working directory
WORKDIR /app

# 4️⃣ Copy dependency files first for caching
COPY pyproject.toml poetry.lock* /app/

# 5️⃣ Install pip and Poetry (safe version)
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install poetry==1.8.6

# 6️⃣ Tell Poetry to install directly into system Python (no virtualenv)
RUN poetry config virtualenvs.create false

# 7️⃣ Install Python dependencies from pyproject.toml
RUN poetry install --no-root --no-interaction --no-ansi

# 8️⃣ Copy the rest of your app code
COPY . /app

# 9️⃣ Expose the port for FastAPI
EXPOSE 8080

# 🔟 Start FastAPI
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]