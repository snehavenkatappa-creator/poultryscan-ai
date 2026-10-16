"""Script to prepare a clean, verified 5-class poultry disease dataset.

Classes supported:
1. Coccidiosis
2. Fowl Pox
3. Healthy
4. Newcastle Disease
5. Salmonella

Sources:
- Coccidiosis, Healthy, Newcastle Disease, Salmonella: Labeled poultry pathology dataset.
- Fowl Pox: Authentic veterinary clinical dataset (Evet-Africa / HuggingFace, AGPL-3.0).

Splits:
- dataset/ (training and validation with zero leakage)
- test_dataset/ (unbiased held-out test evaluation)
"""

import os
import shutil
import random
from PIL import Image

random.seed(42)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TEST_DATASET_DIR = os.path.join(BASE_DIR, "test_dataset")

ARCHIVE_BASE = r"C:\Users\sneha\Downloads\archive (1)\poultry_diseases"
FOWLPOX_SRC = os.path.join(BASE_DIR, "dataset_download", "fowlpox")

CLASSES = [
    "Coccidiosis",
    "Fowl Pox",
    "Healthy",
    "Newcastle Disease",
    "Salmonella"
]

def verify_and_copy(src_path, dst_path):
    """Verify image decodability before copying."""
    try:
        with Image.open(src_path) as img:
            img.verify()
        shutil.copyfile(src_path, dst_path)
        return True
    except Exception as e:
        print(f"Skipping corrupt image {src_path}: {e}")
        return False

def main():
    print("Preparing 5-class poultry dataset...")
    for c in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, c), exist_ok=True)
        os.makedirs(os.path.join(TEST_DATASET_DIR, c), exist_ok=True)

    # 1. Fowl Pox
    print("\n[1/5] Processing Fowl Pox images...")
    fp_files = sorted([f for f in os.listdir(FOWLPOX_SRC) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    random.shuffle(fp_files)
    test_fp = fp_files[:30]
    train_fp = fp_files[30:]
    
    for f in test_fp:
        verify_and_copy(os.path.join(FOWLPOX_SRC, f), os.path.join(TEST_DATASET_DIR, "Fowl Pox", f))
    for f in train_fp:
        verify_and_copy(os.path.join(FOWLPOX_SRC, f), os.path.join(DATASET_DIR, "Fowl Pox", f))
    print(f"  Fowl Pox: {len(train_fp)} train/val samples, {len(test_fp)} test samples.")

    # 2. Newcastle Disease
    print("\n[2/5] Processing Newcastle Disease images...")
    ncd_test_src = os.path.join(ARCHIVE_BASE, "test", "ncd")
    ncd_test_files = sorted(os.listdir(ncd_test_src))
    for f in ncd_test_files:
        verify_and_copy(os.path.join(ncd_test_src, f), os.path.join(TEST_DATASET_DIR, "Newcastle Disease", f))

    ncd_train_src = os.path.join(ARCHIVE_BASE, "train", "ncd")
    ncd_valid_src = os.path.join(ARCHIVE_BASE, "valid", "ncd")
    ncd_train_files = [os.path.join(ncd_train_src, f) for f in os.listdir(ncd_train_src)] + \
                      [os.path.join(ncd_valid_src, f) for f in os.listdir(ncd_valid_src)]
    for fp in ncd_train_files:
        fname = os.path.basename(fp)
        verify_and_copy(fp, os.path.join(DATASET_DIR, "Newcastle Disease", fname))
    print(f"  Newcastle Disease: {len(ncd_train_files)} train/val samples, {len(ncd_test_files)} test samples.")

    # 3. Coccidiosis
    print("\n[3/5] Processing Coccidiosis images...")
    cocci_test_src = os.path.join(ARCHIVE_BASE, "test", "cocci")
    cocci_test_files = sorted(os.listdir(cocci_test_src))[:60]
    for f in cocci_test_files:
        verify_and_copy(os.path.join(cocci_test_src, f), os.path.join(TEST_DATASET_DIR, "Coccidiosis", f))

    cocci_train_src = os.path.join(ARCHIVE_BASE, "train", "cocci")
    cocci_train_files = sorted(os.listdir(cocci_train_src))
    random.shuffle(cocci_train_files)
    cocci_selected = cocci_train_files[:450]
    for f in cocci_selected:
        verify_and_copy(os.path.join(cocci_train_src, f), os.path.join(DATASET_DIR, "Coccidiosis", f))
    print(f"  Coccidiosis: {len(cocci_selected)} train/val samples, {len(cocci_test_files)} test samples.")

    # 4. Healthy
    print("\n[4/5] Processing Healthy images...")
    healthy_test_src = os.path.join(ARCHIVE_BASE, "test", "healthy")
    healthy_test_files = sorted(os.listdir(healthy_test_src))[:60]
    for f in healthy_test_files:
        verify_and_copy(os.path.join(healthy_test_src, f), os.path.join(TEST_DATASET_DIR, "Healthy", f))

    healthy_train_src = os.path.join(ARCHIVE_BASE, "train", "healthy")
    healthy_train_files = sorted(os.listdir(healthy_train_src))
    random.shuffle(healthy_train_files)
    healthy_selected = healthy_train_files[:450]
    for f in healthy_selected:
        verify_and_copy(os.path.join(healthy_train_src, f), os.path.join(DATASET_DIR, "Healthy", f))
    print(f"  Healthy: {len(healthy_selected)} train/val samples, {len(healthy_test_files)} test samples.")

    # 5. Salmonella
    print("\n[5/5] Processing Salmonella images...")
    salmo_test_src = os.path.join(ARCHIVE_BASE, "test", "salmo")
    salmo_test_files = sorted(os.listdir(salmo_test_src))[:60]
    for f in salmo_test_files:
        verify_and_copy(os.path.join(salmo_test_src, f), os.path.join(TEST_DATASET_DIR, "Salmonella", f))

    salmo_train_src = os.path.join(ARCHIVE_BASE, "train", "salmo")
    salmo_train_files = sorted(os.listdir(salmo_train_src))
    random.shuffle(salmo_train_files)
    salmo_selected = salmo_train_files[:450]
    for f in salmo_selected:
        verify_and_copy(os.path.join(salmo_train_src, f), os.path.join(DATASET_DIR, "Salmonella", f))
    print(f"  Salmonella: {len(salmo_selected)} train/val samples, {len(salmo_test_files)} test samples.")

    print("\n" + "=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    tot_train = 0
    tot_test = 0
    for c in CLASSES:
        n_tr = len(os.listdir(os.path.join(DATASET_DIR, c)))
        n_te = len(os.listdir(os.path.join(TEST_DATASET_DIR, c)))
        tot_train += n_tr
        tot_test += n_te
        print(f"  {c:20s}: {n_tr:4d} train/val | {n_te:4d} test")
    print("-" * 60)
    print(f"  {'TOTAL':20s}: {tot_train:4d} train/val | {tot_test:4d} test (Grand Total: {tot_train + tot_test})")
    print("=" * 60)

if __name__ == "__main__":
    main()
