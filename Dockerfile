FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY . .

ENV HOME=/app

# Render sets $PORT at runtime (defaults to 10000). Bind gunicorn to it.
# The shell form lets $PORT expand; fallback to 10000 if unset (local runs).
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 2 --timeout 120 app:app