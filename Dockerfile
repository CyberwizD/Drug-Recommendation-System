# Dockerfile for Render deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by Reflex and ML libraries
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models database

# Initialize Reflex (downloads Bun and frontend dependencies)
RUN reflex init

# Expose port
EXPOSE 8000

# Run the application
CMD reflex run --env prod --backend-host 0.0.0.0 --backend-port $PORT
