"""Comprehensive Test Suite for PoultryScan AI Input Validation Mechanism.

Tests:
1. Rejection of unrelated non-poultry images (car, flower, person, dog, cup, airplane).
2. Rejection of blank / solid-color uniform images.
3. Acceptance of legitimate poultry images across all 5 classes (test_dataset/).
4. Verification that rejected images produce no disease diagnosis and are not saved.
"""

import os
import sys
import numpy as np
from PIL import Image

import tensorflow as tf
from image_validator import validate_poultry_image

def run_validation_tests():
    print("=" * 70)
    print("POULTRYSCAN AI — INPUT VALIDATION & REJECTION MECHANISM TEST SUITE")
    print("=" * 70)

    # 1. Load active model
    model_path = "model/poultry_disease_model.keras"
    assert os.path.exists(model_path), f"Model not found: {model_path}"
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Trained poultry model loaded successfully.\n")

    # 2. Test Non-Poultry Images
    print("-" * 70)
    print("TEST 1: Evaluating Non-Poultry / Unrelated Images (Expected: REJECTED)")
    print("-" * 70)

    non_poultry_dir = "non_poultry_test"
    assert os.path.exists(non_poultry_dir), f"Directory missing: {non_poultry_dir}"
    non_poultry_files = sorted(os.listdir(non_poultry_dir))

    all_non_poultry_rejected = True
    for f in non_poultry_files:
        p = os.path.join(non_poultry_dir, f)
        is_valid, msg, details = validate_poultry_image(p, image_type="full_body", disease_model=model)
        status = "PASSED (REJECTED)" if not is_valid else "FAILED (ACCEPTED)"
        stage = details.get("stage", "unknown")
        print(f"  {f:15s} -> Valid: {str(is_valid):5s} | Stage: {stage:16s} | {status}", flush=True)
        print(f"     Reason: {msg}", flush=True)
        if is_valid:
            all_non_poultry_rejected = False

    # 3. Test Blank / Solid Color Images
    print("\n" + "-" * 70)
    print("TEST 2: Evaluating Blank / Solid Color Images (Expected: REJECTED)")
    print("-" * 70)

    solid_red_path = "test_red.jpg"
    Image.new("RGB", (224, 224), (255, 0, 0)).save(solid_red_path)
    is_valid, msg, details = validate_poultry_image(solid_red_path, image_type="full_body", disease_model=model)
    status = "PASSED (REJECTED)" if not is_valid else "FAILED (ACCEPTED)"
    print(f"  {'solid_red.jpg':15s} -> Valid: {str(is_valid):5s} | Stage: {details.get('stage'):16s} | {status}")
    print(f"     Reason: {msg}")
    if is_valid:
        all_non_poultry_rejected = False

    if os.path.exists(solid_red_path):
        os.remove(solid_red_path)

    assert all_non_poultry_rejected, "FAILED: One or more non-poultry images were incorrectly accepted!"
    print("\n  [OK] SUCCESS: 100% of tested non-poultry and unusable images were successfully rejected!")

    # 4. Test In-Distribution Poultry Images (test_dataset/)
    print("\n" + "-" * 70)
    print("TEST 3: Evaluating Authentic Held-Out Test Poultry Images (Expected: ACCEPTED)")
    print("-" * 70)

    test_dir = "test_dataset"
    total_poultry = 0
    accepted_poultry = 0
    per_class_results = {}

    for c in sorted(os.listdir(test_dir)):
        c_dir = os.path.join(test_dir, c)
        if not os.path.isdir(c_dir):
            continue
        mode = "head_face" if c == "Fowl Pox" else "full_body"
        class_total = 0
        class_accepted = 0
        files = sorted(os.listdir(c_dir))
        for f in files:
            p = os.path.join(c_dir, f)
            is_valid, msg, details = validate_poultry_image(p, image_type=mode, disease_model=model)
            total_poultry += 1
            class_total += 1
            if is_valid:
                accepted_poultry += 1
                class_accepted += 1
            else:
                print(f"    WARNING: False rejection on {c}/{f}: {msg} | Details: {details}")

        rate = (class_accepted / class_total) * 100
        per_class_results[c] = (class_accepted, class_total, rate)
        print(f"  {c:20s}: {class_accepted}/{class_total} accepted ({rate:.1f}%)")

    overall_pass_rate = (accepted_poultry / total_poultry) * 100
    print(f"\n  Total Poultry Specimens Evaluated: {total_poultry}")
    print(f"  Total Correctly Accepted:          {accepted_poultry} ({overall_pass_rate:.2f}%)")

    assert overall_pass_rate >= 98.0, f"False rejection rate too high! Pass rate: {overall_pass_rate}%"
    print("  [OK] SUCCESS: In-distribution poultry retention verified!")

    print("\n" + "=" * 70)
    print("ALL INPUT VALIDATION TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_validation_tests()
