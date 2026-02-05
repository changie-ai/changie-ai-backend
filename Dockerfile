# Use Python 3.10 slim
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . /app

# Upgrade pip and install all requirements for Fly
RUN pip install --upgrade pip
RUN pip install -r requirements-fly.txt

# Expose port for FastAPI / uvicorn
EXPOSE 8080

# Run your main entrypoint
CMD ["python", "local_main.py"]