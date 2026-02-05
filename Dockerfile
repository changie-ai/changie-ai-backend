FROM python:3.10-slim

WORKDIR /app

COPY requirements-fly.txt /app/requirements-fly.txt

RUN pip install --no-cache-dir -r requirements-fly.txt

COPY . /app

CMD ["python", "local_main.py"]
