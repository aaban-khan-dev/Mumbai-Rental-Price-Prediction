# Slim Python base — Linux, matches every free host
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs some ML wheels (xgboost, lightgbm, shap/llvmlite) expect at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps  (this layer is cached unless requirements change)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy the rest of the project (respecting .dockerignore)
COPY . .

# The model.pkl gets downloaded here at runtime
ENV HOME=/app

# Flask app listens on 5000
EXPOSE 5000

# Production server. app:app  ->  file app.py, Flask object named `app`.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]