# Dockerfile for Render deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models database

# Expose port
EXPOSE 8000

# Run the application
CMD reflex run --env prod --backend-only & reflex run --env prod --frontend-only --backend-port 8000
