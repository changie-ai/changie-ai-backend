FROM python:3.11-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=120

COPY requirements-fly.txt /app/requirements-fly.txt

# DO NOT upgrade pip or setuptools on Fly
RUN pip install \
    --retries 15 \
    --timeout 120 \
    --prefer-binary \
    -r requirements-fly.txt

COPY . /app

CMD ["python", "main.py"]