FROM python:3.10-slim

WORKDIR /app

COPY wheels /wheels
COPY requirements-fly.txt /app/requirements-fly.txt

RUN pip install --no-index --find-links=/wheels -r requirements-fly.txt

COPY . /app

CMD ["python", "main.py"]