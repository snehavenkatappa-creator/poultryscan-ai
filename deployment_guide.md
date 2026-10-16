# PoultryScan AI — Deployment & Production Architecture Guide

This document assesses cloud and on-premise production deployment compatibility for PoultryScan AI.

---

## 1. Resource & Sizing Assessment

| Component | Disk Footprint | Peak RAM Usage | Model Engine |
| :--- | :---: | :---: | :---: |
| **5-Class Disease Classifier** | 22.7 MB | ~350 MB | Keras 3 / MobileNetV2 |
| **COCO Specimen Detector** | 74.2 MB | ~450 MB | PyTorch CPU / FasterRCNN-MobileNetV3 |
| **ImageNet Semantic Filter** | 14.0 MB | ~200 MB | Keras 3 / MobileNetV2 |
| **Flask + Gunicorn Web App** | < 10 MB | ~150 MB | Python 3.11 WSGI |
| **Total System Footprint** | **~120 MB** | **~900 MB – 1.2 GB** | **Zero GPU required (CPU-optimized)** |

### Hardware Requirements
- **Minimum:** 1 vCPU, 1.5 GB RAM (e.g. AWS `t3.small`, Render Starter, Railway Developer).
- **Recommended:** 2 vCPU, 2.0 GB RAM.
- **Not Recommended:** 512 MB Free-Tier tiers (e.g. Render Free Tier) will encounter Out-Of-Memory (OOM) during TensorFlow and PyTorch joint cold-start import.

---

## 2. Windows Local Launcher (`run_poultryscan.bat`)

Double-clicking `run_poultryscan.bat` provides instant local execution:
1. **Duplicate Prevention:** Checks whether port 5000 is already active (`netstat -ano | findstr :5000`). If already running, opens the browser immediately without spawning duplicate processes.
2. **Environment Resolution:** Automatically detects Anaconda Python (`C:\Users\sneha\anaconda3\python.exe`), active Conda virtual environment, or system Python from `PATH`.
3. **Dependency Diagnostics:** Verifies that Flask, TensorFlow, Keras, PyTorch, Torchvision, and Pillow are installed.
4. **Auto-Browser Launch:** Opens `http://localhost:5000` automatically in your default browser.
5. **Session Persistence:** Keeps the console open and server logs visible until stopped with `Ctrl+C`. Works after laptop reboot without requiring external tools.

---

## 3. Docker Deployment (Recommended for Cloud)

A production [`Dockerfile`](file:///c:/Users/sneha/OneDrive/Documents/Chicken-Disease-Detection-2/Chicken-Disease-Detection%2022222/Dockerfile) is included in the project root.

### Build and Run Locally with Docker:
```bash
# 1. Build Docker image
docker build -t poultryscan-ai:latest .

# 2. Run container on port 5000 with 2GB memory limit
docker run -d -p 5000:5000 --memory="2g" --name poultryscan poultryscan-ai:latest

# 3. Access in browser
http://localhost:5000
```

---

## 4. Cloud Platform Setup (Render / Railway / Fly.io)

### Option A: Railway
1. Push this repository to GitHub.
2. Log in to [Railway.app](https://railway.app/) and click **New Project** &rarr; **Deploy from GitHub repo**.
3. Railway automatically detects `Dockerfile` or `Procfile`.
4. Set Environment Variables:
   - `SECRET_KEY`: `<your-random-secret-key>`
   - `PORT`: `5000`
5. Railway provides a public HTTPS domain (e.g., `https://poultryscan-production.up.railway.app`).

### Option B: Render
1. Create a **Web Service** on [Render.com](https://render.com/).
2. Select **Docker** environment.
3. Select the **Starter Plan** (2 GB RAM) to ensure smooth PyTorch + TensorFlow cold start.
4. Set Health Check path to `/health`.

---

## 5. Security & Persistence Best Practices
- **Persistent Volume:** If deploying to container services with ephemeral file systems, mount a persistent volume at `/app/database` and `/app/uploads` so scan history and uploaded images persist across restarts.
- **WSGI Concurrency:** Set Gunicorn workers to `1` with `--threads 4` to minimize duplicate model weight memory overhead while enabling concurrent request handling.
