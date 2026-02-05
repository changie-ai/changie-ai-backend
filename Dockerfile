FROM python:3.11-slim

WORKDIR /app

# Improve pip reliability on Fly
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements-fly.txt /app/requirements-fly.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install \
      --retries 10 \
      --timeout 120 \
      --index-url https://pypi.org/simple \
      -r requirements-fly.txt

COPY . /app

CMD ["python", "main.py"]