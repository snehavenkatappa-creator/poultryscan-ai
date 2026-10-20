"""PoultryScan AI — Automated Hugging Face Spaces Deployment Script.

This script deploys the production PoultryScan AI Docker application directly to
Hugging Face Spaces using the official huggingface_hub Python SDK.

It handles:
1. Authentication via Hugging Face User Access Token (read from HF_TOKEN env or interactive input).
2. Space creation (or validation) with space_sdk='docker'.
3. Direct file sync respecting .dockerignore (excludes ~1.8GB training/test images, keeps model weights & web assets).
4. Monitoring build logs and testing the live public URL.
"""

import os
import sys
import time
import urllib.request
import urllib.error
import getpass

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("ERROR: huggingface_hub is not installed. Install it with: pip install huggingface_hub")
    sys.exit(1)

REPO_ID = os.environ.get("HF_SPACE_ID", "snehavenkatappa/poultryscan-ai")
PUBLIC_URL = f"https://{REPO_ID.replace('/', '-').lower()}.hf.space"
EMBED_URL = f"https://huggingface.co/spaces/{REPO_ID}"

def main():
    print("=" * 75)
    print("POULTRYSCAN AI — HUGGING FACE SPACES CLOUD DEPLOYMENT")
    print("=" * 75)
    print(f"Target Space:     {REPO_ID}")
    print(f"Permanent URL:    {PUBLIC_URL}")
    print(f"HuggingFace View: {EMBED_URL}")
    print("=" * 75)

    # 1. Acquire Token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nPlease enter your Hugging Face User Access Token (with 'write' permission):")
        print("(You can create one for free at https://huggingface.co/settings/tokens)")
        token = getpass.getpass("Token (hidden input): ").strip()
        if not token:
            print("Token was empty. Exiting.")
            sys.exit(1)

    api = HfApi(token=token)

    # Verify credentials
    try:
        user_info = api.whoami()
        username = user_info.get("name", "Unknown")
        print(f"\n[OK] Authenticated successfully as Hugging Face user: {username}")
    except Exception as exc:
        print(f"\n[ERROR] Authentication failed: {exc}")
        print("Please ensure your token is valid and has 'write' permissions.")
        sys.exit(1)

    # 2. Create or verify Space
    print(f"\n[Step 1] Ensuring Space '{REPO_ID}' exists with Docker SDK...")
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False
        )
        print(f"  [OK] Space '{REPO_ID}' is ready.")
    except Exception as exc:
        print(f"  Note while checking space: {exc}")

    # 3. Synchronize files
    print("\n[Step 2] Uploading application files, model weights, and assets...")
    print("  (Excluding dataset/, test_dataset/, and local training scratch files)")

    ignore_list = [
        "dataset/**",
        "dataset",
        "test_dataset/**",
        "test_dataset",
        "dataset_download/**",
        "dataset_download",
        "non_poultry_test/**",
        "non_poultry_test",
        "backups_stage8/**",
        "backups_stage8",
        "uploads/*",
        "!uploads/.gitkeep",
        "database/*.db",
        ".git/**",
        ".git",
        "venv/**",
        ".venv/**",
        "__pycache__/**",
        "*.pyc",
        "*.backup",
        "app_server.log",
        "train_model.py",
        "train_efficientnet.py",
        "prepare_dataset.py",
        "evaluate_model.py",
        "inspect_model.py",
        "test_*.py",
        ".env",
        "*.env",
        "deploy_to_hf.py"
    ]

    try:
        commit_info = api.upload_folder(
            folder_path=".",
            repo_id=REPO_ID,
            repo_type="space",
            ignore_patterns=ignore_list,
            commit_message="Deploy PoultryScan AI v2.1 production Docker application"
        )
        print("  [OK] Files uploaded successfully.")
        print(f"  Commit URL: {commit_info}")
    except Exception as exc:
        print(f"  [ERROR] File upload failed: {exc}")
        sys.exit(1)

    # 4. Monitor build
    print("\n[Step 3] Monitoring Space build and runtime status...")
    print("  Hugging Face is building your Docker image with 16 GB RAM...")
    print("  This typically takes 2 to 4 minutes on the first build.\n")

    max_checks = 60
    for i in range(max_checks):
        try:
            runtime = api.get_space_runtime(repo_id=REPO_ID)
            stage = runtime.stage
            hardware = runtime.hardware or "cpu-basic"
            print(f"  [{i+1}/{max_checks}] Status: {stage} (Hardware: {hardware})")

            if stage == "RUNNING":
                print("\n" + "=" * 75)
                print("SUCCESS: PoultryScan AI is RUNNING live in the cloud!")
                print("=" * 75)
                break
            elif stage in ("BUILD_ERROR", "RUNTIME_ERROR"):
                print(f"\n[ALERT] Build/Runtime encountered an issue: {stage}")
                print(f"Check build logs at: {EMBED_URL}")
                break
        except Exception as exc:
            print(f"  Checking status... ({exc})")

        time.sleep(10)

    # 5. Live Health Check
    print(f"\n[Step 4] Testing live HTTP endpoint at {PUBLIC_URL}...")
    try:
        req = urllib.request.Request(
            f"{PUBLIC_URL}/",
            headers={"User-Agent": "PoultryScan-Verifier/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            code = resp.getcode()
            print(f"  [PASSED] HTTP GET / returned status code: {code}")
            print(f"\n  Website is fully operational and publicly accessible!")
            print(f"  Direct Link:  {PUBLIC_URL}")
            print(f"  Space Link:   {EMBED_URL}")
    except Exception as exc:
        print(f"  Status check note: {exc}")
        print(f"  If the container is still finishing cold start, please open {PUBLIC_URL} in your browser in ~1-2 minutes.")

if __name__ == "__main__":
    main()
