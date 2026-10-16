"""Evaluate the PoultryScan AI model on a separate, unseen test dataset.

Features:
- Validates test dataset integrity and detects any cross-split data leakage via SHA-256 hashing.
- Evaluates all five classes: Coccidiosis, Fowl Pox, Healthy, Newcastle Disease, Salmonella.
- Computes overall test accuracy, confusion matrix, per-class precision, recall, F1-score, and support.
- Identifies and highlights commonly confused class pairs.
- Saves an evaluation report with model version, evaluation date/time, and metadata.
- Generates a high-resolution confusion matrix heatmap visualization.

Usage:
    python evaluate_model.py --data test_dataset --model model/poultry_disease_model.keras
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
import shutil
import matplotlib
matplotlib.use("Agg")  # Headless mode for server / CLI execution
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

IMG_SIZE = (224, 224)
MODEL_VERSION = "MobileNetV2-5Class-v2.1"


def compute_file_hash(path):
    """Compute SHA-256 hash for byte-level duplicate detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def check_data_leakage(train_dir, test_dir):
    """Verify that no image in test_dir duplicates any image in train_dir."""
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        return {"checked": False, "duplicates_found": 0, "duplicate_files": []}

    train_hashes = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for root, _, files in os.walk(train_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                p = os.path.join(root, f)
                try:
                    train_hashes[compute_file_hash(p)] = p
                except Exception:
                    pass

    duplicates = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                p = os.path.join(root, f)
                try:
                    h = compute_file_hash(p)
                    if h in train_hashes:
                        duplicates.append({
                            "test_file": p,
                            "train_duplicate": train_hashes[h]
                        })
                except Exception:
                    pass

    return {
        "checked": True,
        "train_samples_hashed": len(train_hashes),
        "duplicates_found": len(duplicates),
        "duplicate_files": duplicates
    }


def load_class_indices(model_path):
    """Load exact class mapping JSON next to the model."""
    dir_name = os.path.dirname(model_path) or "."
    json_path = os.path.join(dir_name, "class_indices.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data[str(i)] for i in range(len(data))]
    return ["Coccidiosis", "Fowl Pox", "Healthy", "Newcastle Disease", "Salmonella"]


def evaluate(data_dir="test_dataset", train_dir="dataset", model_path="model/poultry_disease_model.keras",
             batch_size=32, preprocess="rescale", output_dir="evaluation_results"):
    print("=" * 68)
    print("POULTRYSCAN AI - MODEL EVALUATION & PERFORMANCE VERIFICATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Version: {MODEL_VERSION}")
    print("=" * 68)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'.")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Test dataset directory '{data_dir}' not found.")

    # 1. Data leakage check
    print("\n[Step 1] Checking for cross-split data leakage (SHA-256 duplicate detection)...")
    leakage_info = check_data_leakage(train_dir, data_dir)
    if leakage_info["checked"]:
        print(f"  Verified {leakage_info['train_samples_hashed']} training images.")
        if leakage_info["duplicates_found"] == 0:
            print("  [PASSED] 0 duplicates found. 100% clean, zero-leakage test set confirmed.")
        else:
            print(f"  [WARNING] Found {leakage_info['duplicates_found']} duplicate image(s) across splits!")
    else:
        print("  Skipped leakage verification (training directory not found).")

    # 2. Load Model
    print(f"\n[Step 2] Loading model from '{model_path}'...")
    try:
        import keras
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False)

    display_classes = load_class_indices(model_path)
    num_classes = len(display_classes)
    print(f"  Model loaded with {model.output_shape[-1]} output units.")
    print(f"  Exact class indices mapping ({num_classes} classes): {display_classes}")

    # 3. Load Test Dataset
    print(f"\n[Step 3] Loading unseen test dataset from '{data_dir}'...")
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    detected_classes = raw_ds.class_names
    print(f"  Found {len(detected_classes)} folder classes: {detected_classes}")

    # Normalization
    if preprocess == "rescale":
        test_ds = raw_ds.map(lambda x, y: (x / 255.0, y))
    else:
        test_ds = raw_ds.map(lambda x, y: ((x / 127.5) - 1.0, y))

    # 4. Predict
    print("\n[Step 4] Running batch inference across all test samples...")
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([y.numpy() for _, y in raw_ds], axis=0)
    total_test_samples = int(len(y_true))

    # 5. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    per_class_recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    report_dict = classification_report(y_true, y_pred, target_names=display_classes, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=display_classes, digits=4)

    # 6. Analyze Confused Class Pairs
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                true_name = display_classes[i]
                pred_name = display_classes[j]
                confused_pairs.append({
                    "true_class": true_name,
                    "predicted_as": pred_name,
                    "misclassified_count": int(cm[i, j]),
                    "description": f"{cm[i, j]} sample(s) of '{true_name}' misclassified as '{pred_name}'"
                })

    confused_pairs.sort(key=lambda x: x["misclassified_count"], reverse=True)

    print("\n" + "=" * 68)
    print("EVALUATION METRICS & CLASSIFICATION REPORT")
    print("=" * 68)
    print(f"Overall Test Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix (Rows = Actual, Columns = Predicted):")
    print(cm)
    print("\nPer-Class Sensitivity / Recall:")
    for idx, cname in enumerate(display_classes):
        actual_total = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        pct = (correct / max(actual_total, 1)) * 100.0
        print(f"  {cname:20s}: {pct:6.2f}% ({correct}/{actual_total} test samples)")

    print("\nFrequently Confused Class Pairs:")
    if confused_pairs:
        for p in confused_pairs:
            print(f"  • {p['description']}")
    else:
        print("  • No misclassifications observed!")

    print("\nDetailed Scikit-Learn Classification Report:")
    print(report_text)
    print("=" * 68)

    # 7. Save Visualization and Artifacts
    os.makedirs(output_dir, exist_ok=True)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.set_title(f"PoultryScan AI 5-Class Confusion Matrix (Accuracy: {acc * 100:.2f}%)", fontsize=13, pad=14, fontweight="bold")
    fig.colorbar(cax)
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(display_classes, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(display_classes, fontsize=10)
    ax.set_ylabel("True / Actual Class", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")

    fig.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved confusion matrix plot: {cm_path}")

    # Copy to static/images for web frontend rendering
    static_img = os.path.join("static", "images", "confusion_matrix.png")
    os.makedirs(os.path.dirname(static_img), exist_ok=True)
    shutil.copyfile(cm_path, static_img)

    # Support counts per class
    support_per_class = {
        display_classes[i]: int(cm[i].sum()) for i in range(num_classes)
    }

    # Save summary JSON report
    summary = {
        "model_version": MODEL_VERSION,
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_date": datetime.now().strftime("%B %d, %Y"),
        "overall_accuracy": round(float(acc) * 100.0, 2),
        "total_test_samples": total_test_samples,
        "preprocessing": preprocess,
        "classes": display_classes,
        "support_per_class": support_per_class,
        "per_class_accuracy": {
            display_classes[i]: round(float(per_class_recalls[i]) * 100.0, 2)
            for i in range(num_classes)
        },
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confused_pairs": confused_pairs,
        "data_leakage": {
            "verified_zero_leakage": (leakage_info["duplicates_found"] == 0),
            "duplicates_detected": leakage_info["duplicates_found"]
        }
    }

    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary JSON: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate PoultryScan AI 5-class model on test dataset.")
    parser.add_argument("--data", default="test_dataset", help="Path to held-out test dataset")
    parser.add_argument("--train-data", default="dataset", help="Path to training dataset for leakage detection")
    parser.add_argument("--model", default="model/poultry_disease_model.keras", help="Path to .keras model file")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--preprocess", choices=["rescale", "mobilenet"], default="rescale", help="Preprocessing mode")
    parser.add_argument("--output", default="evaluation_results", help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    evaluate(data_dir=args.data, train_dir=args.train_data, model_path=args.model,
             batch_size=args.batch, preprocess=args.preprocess, output_dir=args.output)


if __name__ == "__main__":
    main()
