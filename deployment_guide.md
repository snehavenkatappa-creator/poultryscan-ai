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

## 3. Hugging Face Spaces Deployment (Permanent Cloud Hosting)

Hugging Face Spaces provides 16 GB RAM and 2 vCPUs on its free tier, making it the ideal host for PoultryScan AI's joint TensorFlow (MobileNetV2) and PyTorch (FasterRCNN) CPU inference architecture.

### Public URLs for Your Deployed App:
* **Direct Web App URL:** `https://snehavenkatappa-poultryscan-ai.hf.space`
* **Hugging Face Space Embed:** `https://huggingface.co/spaces/snehavenkatappa/poultryscan-ai`

### Automated 1-Step Deployment via Python:
Run the provided automated deployment script:
```powershell
python deploy_to_hf.py
```
This script will prompt you for your Hugging Face User Access Token (create one for free with **write** role at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)), upload the production code and model weights (ignoring training data), and verify the live public URL.

### Git-Based Deployment:
Alternatively, you can push directly to Hugging Face Spaces via Git:
```powershell
# 1. Add Hugging Face Space git remote (already configured)
git remote add space https://huggingface.co/spaces/snehavenkatappa/poultryscan-ai

# 2. Push to the Space main branch
git push space main
```
*When prompted for credentials by Git, use your Hugging Face username as Username, and your Hugging Face Access Token as Password.*

### Production Architecture on Hugging Face:
1. **Container Port:** Gunicorn binds to `0.0.0.0:7860` (the standard HF Spaces port configured in [`Dockerfile`](file:///c:/Users/sneha/OneDrive/Documents/Chicken-Disease-Detection-2/Chicken-Disease-Detection%2022222/Dockerfile) and [`app.py`](file:///c:/Users/sneha/OneDrive/Documents/Chicken-Disease-Detection-2/Chicken-Disease-Detection%2022222/app.py)).
2. **Persistent Storage:** The application automatically detects Hugging Face's persistent `/data` volume and stores uploaded images in `/data/uploads` and SQLite scan history in `/data/database/predictions.db` so records survive container rebuilds.
3. **Usage & Sleep Limits:**
   * **Free Tier Inactivity Sleep:** If no visits occur for 48 consecutive hours, Hugging Face enters an idle sleep state to conserve resources.
   * **Automatic Wake-Up:** The moment any visitor accesses `https://snehavenkatappa-poultryscan-ai.hf.space` from a phone or PC, the container automatically wakes up within ~10–15 seconds.
   * **Bandwidth & Requests:** Unlimited HTTP/HTTPS requests with no monthly bandwidth charge on the free tier.

---

## 4. Docker Deployment (Local or Alternative Cloud)

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

## 5. Alternative Cloud Platforms (Render / Railway)

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

## 6. Security & Persistence Best Practices
- **Persistent Volume:** If deploying to container services with ephemeral file systems, mount a persistent volume at `/app/database` and `/app/uploads` so scan history and uploaded images persist across restarts.
- **WSGI Concurrency:** Set Gunicorn workers to `1` with `--threads 4` to minimize duplicate model weight memory overhead while enabling concurrent request handling.

