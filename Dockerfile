FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY ml/ ml/

# Create output and data directories
RUN mkdir -p artifacts/models output/explain output/forecast data

# Default entrypoint
ENTRYPOINT ["python", "-m", "ml.cli"]
CMD ["train"]
