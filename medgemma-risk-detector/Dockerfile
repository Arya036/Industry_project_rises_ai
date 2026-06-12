FROM python:3.10-slim

# Install system dependencies for OpenCV and graphic operations
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy python dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy configs and local directories
COPY config.yaml .
COPY /models ./models
COPY /src ./src

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
