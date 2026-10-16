# ============================================================
# PoultryScan AI — Production Dockerfile
# Target: Hugging Face Spaces (Docker) — 16 GB RAM free tier
# Base:   Python 3.11 slim (Debian bookworm)
# ============================================================
FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
# Use CPU-only torch to avoid pulling CUDA dependencies (~5 GB extra)
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# System libs required by OpenCV, Pillow (JPEG/PNG codecs), and font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install CPU-only PyTorch first (saves ~3 GB over the default CUDA wheel) ──
# Pin versions that are known-compatible with torchvision 0.15.x on Python 3.11
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining Python dependencies ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source (model weights, templates, static, etc.) ──
COPY . .

# Create writable directories for uploads and SQLite database.
# On HF Spaces, /data is a persistent volume — we symlink into it at runtime
# via the entrypoint so uploads and DB survive container restarts.
RUN mkdir -p /app/uploads /app/database \
    && chmod -R 777 /app/uploads /app/database \
    && mkdir -p /data/uploads /data/database \
    && chmod -R 777 /data

# HF Spaces runs as user 1000 (non-root). Ensure all files are accessible.
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app /data
USER 1000

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Start with Gunicorn (production WSGI server).
# 1 worker + 4 threads: keeps memory for TF + PyTorch under 2 GB peak.
# Timeout 300 s covers cold-start model loading on first request.
CMD ["sh", "-c", \
     "gunicorn \
      --bind 0.0.0.0:${PORT:-7860} \
      --workers 1 \
      --threads 4 \
      --timeout 300 \
      --keep-alive 5 \
      --log-level info \
      --access-logfile - \
      --error-logfile - \
      app:app"]
