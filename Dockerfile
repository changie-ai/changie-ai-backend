# 1. Use a Python image
FROM python:3.13-slim

# 2. Install system dependencies
RUN apt-get update && apt-get install -y \
    rubberband-cli \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Set work directory
WORKDIR /app

# 4. Copy project files
COPY . .

# 5. Install Python dependencies with Poetry
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install --no-root

# 6. Expose port for FastAPI
EXPOSE 8000

# 7. Run the app
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
