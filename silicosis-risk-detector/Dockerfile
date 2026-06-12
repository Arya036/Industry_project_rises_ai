# =====================================================================
# Silicosis Detection System — Production Container
# =====================================================================
# NOTE: This container runs the binary classifier and finding classifier
# locally via ONNX Runtime (CPU mode).
#
# The MedGemma report generator (4.3B params) is NOT included in this
# container — it runs on Kaggle (T4 GPU) and is called via ngrok URL.
# Set the ngrok_url in config.yaml before starting the server.
# =====================================================================

FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (Docker cache layer optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy configuration and source code
COPY config.yaml .
COPY /models ./models
COPY /src ./src

# Port exposed by FastAPI server
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
