# Multi-stage lightweight production container
FROM python:3.11-slim as base

WORKDIR /app

# Prevent Python from writing .pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run validation tests during build
RUN python tests/run_tests.py

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
