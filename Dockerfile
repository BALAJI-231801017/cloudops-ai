FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application source code
COPY . .

# Pre-train Isolation Forest baseline model during container image build
RUN python -m backend.ml.train

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV FLASK_DEBUG=0

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

CMD ["python", "-m", "backend.app"]
