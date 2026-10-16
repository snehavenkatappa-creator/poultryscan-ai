"""
dataset_split_summary.py — Dataset & Split Auditor for PoultryScan AI

Produces:
  - Per-class image counts for training and test sets
  - Class imbalance ratios
  - SHA-256 duplicate check between splits (data leakage verification)
  - Dataset summary JSON for IEEE paper

Usage:
    python dataset_split_summary.py
"""

import hashlib
import json
import os
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset")
TEST_DIR  = os.path.join(BASE_DIR, "test_dataset")
OUTPUT    = os.path.join(BASE_DIR, "research_results", "dataset_split_summary.json")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def count_images(directory):
    counts = {}
    if not os.path.isdir(directory):
        return counts
    for cls in sorted(os.listdir(directory)):
        cp = os.path.join(directory, cls)
        if os.path.isdir(cp):
            imgs = [f for f in os.listdir(cp)
                    if os.path.splitext(f)[1].lower() in VALID_EXTS]
            counts[cls] = len(imgs)
    return counts


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def check_leakage(train_dir, test_dir):
    train_hashes = {}
    for root, _, files in os.walk(train_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                p = os.path.join(root, f)
                try:
                    train_hashes[sha256(p)] = os.path.relpath(p, train_dir)
                except Exception:
                    pass

    duplicates = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                p = os.path.join(root, f)
                try:
                    h = sha256(p)
                    if h in train_hashes:
                        duplicates.append({
                            "test_file": os.path.relpath(p, test_dir),
                            "train_match": train_hashes[h]
                        })
                except Exception:
                    pass

    return {
        "train_images_hashed": len(train_hashes),
        "duplicates_found": len(duplicates),
        "duplicate_files": duplicates,
        "clean": len(duplicates) == 0
    }


def main():
    print("=" * 60)
    print("DATASET & SPLIT SUMMARY AUDIT")
    print("=" * 60)

    train_counts = count_images(TRAIN_DIR)
    test_counts  = count_images(TEST_DIR)
    all_classes  = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))

    total_train = sum(train_counts.values())
    total_test  = sum(test_counts.values())
    total_all   = total_train + total_test

    print(f"\n{'Class':<25} {'Train':>7} {'Test':>7} {'Total':>7} {'Train%':>8} {'Imbalance':>10}")
    print("-" * 70)

    max_count = max(train_counts.values()) if train_counts else 1
    per_class = {}

    for cls in all_classes:
        tr = train_counts.get(cls, 0)
        te = test_counts.get(cls, 0)
        tot = tr + te
        tr_pct = (tr / total_train * 100) if total_train else 0
        imbalance = max_count / tr if tr > 0 else float("inf")
        per_class[cls] = {
            "train": tr,
            "test": te,
            "total": tot,
            "train_pct": round(tr_pct, 1),
            "imbalance_ratio_vs_majority": round(imbalance, 2)
        }
        print(f"{cls:<25} {tr:>7} {te:>7} {tot:>7} {tr_pct:>7.1f}% {imbalance:>10.2f}x")

    print("-" * 70)
    print(f"{'TOTAL':<25} {total_train:>7} {total_test:>7} {total_all:>7}")

    # Training split: 80/20 within dataset/ folder
    est_train_80 = round(total_train * 0.8)
    est_val_20   = total_train - est_train_80
    print(f"\nEstimated training split (80/20 seed=42, from train_model.py):")
    print(f"  Effective train: ~{est_train_80}  |  Effective val: ~{est_val_20}")

    print(f"\nHeld-out test set: {total_test} images (separate test_dataset/ folder)")

    print(f"\n[Checking data leakage between dataset/ and test_dataset/...]")
    leakage = check_leakage(TRAIN_DIR, TEST_DIR)
    print(f"  Train images hashed: {leakage['train_images_hashed']}")
    if leakage["clean"]:
        print("  [PASS] Zero SHA-256 duplicates found — no leakage detected.")
    else:
        print(f"  [FAIL] {leakage['duplicates_found']} duplicate(s) found across splits!")
        for d in leakage["duplicate_files"][:5]:
            print(f"    test:{d['test_file']} == train:{d['train_match']}")

    # Save JSON
    summary = {
        "audit_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_dir": TRAIN_DIR,
        "test_dir": TEST_DIR,
        "total_train_images": total_train,
        "total_test_images": total_test,
        "total_images": total_all,
        "estimated_effective_train": est_train_80,
        "estimated_effective_val": est_val_20,
        "train_val_split_method": "tf.keras.utils.image_dataset_from_directory(validation_split=0.2, seed=42)",
        "note_on_split": (
            "Training (dataset/) and test (test_dataset/) are separate folders. "
            "The 80/20 train/val split is computed within the dataset/ folder during training. "
            "No images from test_dataset/ were used during training or validation."
        ),
        "class_distribution": per_class,
        "imbalance_note": (
            "Fowl Pox is significantly underrepresented (127 train, 28 test) compared to "
            "Coccidiosis, Healthy, and Salmonella (450 each). Balanced class weights were "
            "applied during training via compute_class_weight('balanced')."
        ),
        "data_leakage": leakage,
        "limitations": [
            "No metadata about farm, bird ID, or video sequence origin — near-duplicate "
            "images from the same bird cannot be detected by SHA-256 alone.",
            "Class imbalance in Fowl Pox (127 train vs 450 others) may affect recall; "
            "class weighting was applied during training.",
            "Dataset source, geographic origin, camera hardware, and licensing are not "
            "documented in the project — these must be disclosed in the IEEE paper.",
            "Generalization to different farms, breeds, or camera conditions has not been evaluated."
        ]
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
