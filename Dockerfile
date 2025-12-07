FROM python:3.9-slim

WORKDIR /app

# Minimal environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# System packages needed for scikit-learn and qdrant-client wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install only essential Python dependencies based on code imports
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scikit-learn==1.3.1 \
    qdrant-client==1.6.3 \
    requests==2.31.0 \
    langchain-community==0.0.13

# Copy project files
COPY services ./services
COPY docs ./docs

# Default command can be overridden by docker-compose
CMD ["python", "-u", "services/test_simple.py"]
