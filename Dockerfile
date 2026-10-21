# ============================================================
# PoultryScan AI — Production Dockerfile
# Target: Render Free Tier (512 MB RAM) / Cloud Containers
# Base:   Python 3.11 slim (Debian bookworm)
# ============================================================
FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Strict single-thread memory control for 512 MB RAM environments
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1

WORKDIR /app

# System libs required by OpenCV, Pillow (JPEG/PNG codecs), and font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (model weights, templates, static, etc.)
COPY . .

# Create writable directories for uploads and SQLite database
RUN mkdir -p /app/uploads /app/database \
    && chmod -R 777 /app/uploads /app/database

# Run as non-root user for security
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER 1000

# Expose Render default port (10000) or fallback
EXPOSE 10000

# Start with Gunicorn (1 worker + 2 threads, with max-requests memory recycling)
CMD ["sh", "-c", \
     "gunicorn \
      --bind 0.0.0.0:${PORT:-10000} \
      --workers 1 \
      --threads 2 \
      --timeout 180 \
      --max-requests 100 \
      --max-requests-jitter 20 \
      --keep-alive 5 \
      --log-level info \
      --access-logfile - \
      --error-logfile - \
      app:app"]
