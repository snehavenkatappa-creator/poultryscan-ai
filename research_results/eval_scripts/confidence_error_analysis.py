"""
confidence_error_analysis.py — Confidence Calibration & Error Analysis
for PoultryScan AI MobileNetV2 5-Class Model

Analyzes:
  - Softmax confidence distributions for correct vs. incorrect predictions
  - High-confidence wrong predictions (miscalibration indicators)
  - Per-class confidence statistics
  - Reliability diagram (calibration curve)
  - Expected Calibration Error (ECE)
  - Brier Score (probabilistic accuracy)

Outputs saved to: research_results/

IMPORTANT:
  - Confidence values are softmax probabilities, NOT logits.
  - High confidence does NOT guarantee correctness.
  - Calibration curves are computed on the TEST set only (not used for threshold fitting).
  - No thresholds are adjusted based on this analysis.

Usage:
    python confidence_error_analysis.py \\
        --model model/poultry_disease_model.keras \\
        --test-dir test_dataset \\
        --output research_results
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize

IMG_SIZE = (224, 224)


def load_class_indices(model_path):
    ci = os.path.join(os.path.dirname(model_path) or ".", "class_indices.json")
    if os.path.exists(ci):
        with open(ci) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data[str(i)] for i in range(len(data))]
    return ["Coccidiosis", "Fowl Pox", "Healthy", "Newcastle Disease", "Salmonella"]


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def compute_ece(y_true, y_prob_pred, n_bins=10):
    """Expected Calibration Error using equal-width bins."""
    confidences = np.max(y_prob_pred, axis=1)
    predictions = np.argmax(y_prob_pred, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += (mask.sum() / n) * abs(avg_acc - avg_conf)

    return ece, bin_boundaries, confidences, accuracies


def run_analysis(model_path, test_dir, output_dir):
    print("=" * 68)
    print("POULTRYSCAN AI — CONFIDENCE & ERROR ANALYSIS")
    print("=" * 68)

    model = tf.keras.models.load_model(model_path, compile=False)
    class_names = load_class_indices(model_path)
    num_classes = len(class_names)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    os.makedirs(output_dir, exist_ok=True)

    # Verify output type
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    dummy_out = model.predict(dummy, verbose=0)[0]
    output_sum = float(np.sum(dummy_out))
    output_type = "softmax_probabilities" if 0.98 < output_sum < 1.02 else "logits_or_other"
    print(f"\nOutput type verification: sum={output_sum:.4f} -> {output_type}")
    if output_type != "softmax_probabilities":
        print("  WARNING: Outputs do not sum to ~1.0. Treat confidence values with caution.")

    # Collect all predictions
    all_y_true, all_y_pred, all_probs = [], [], []
    error_samples = []     # high-confidence wrong predictions
    all_samples = []

    for cls_index, cls_name in enumerate(class_names):
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        images = [f for f in os.listdir(cls_dir)
                  if os.path.splitext(f)[1].lower() in valid_exts]
        for img_file in images:
            img_path = os.path.join(cls_dir, img_file)
            try:
                inp = preprocess(img_path)
                probs = model.predict(inp, verbose=0)[0]
                pred_index = int(np.argmax(probs))
                confidence = float(probs[pred_index])

                all_y_true.append(cls_index)
                all_y_pred.append(pred_index)
                all_probs.append(probs)

                record = {
                    "image": img_path,
                    "gt_class": cls_name,
                    "gt_index": cls_index,
                    "pred_class": class_names[pred_index],
                    "pred_index": pred_index,
                    "confidence_pct": round(confidence * 100, 2),
                    "correct": pred_index == cls_index,
                    "all_probs": {class_names[i]: round(float(probs[i]) * 100, 2)
                                  for i in range(num_classes)}
                }
                all_samples.append(record)

                if not (pred_index == cls_index) and confidence >= 0.80:
                    error_samples.append(record)

            except Exception as exc:
                print(f"  [ERROR] {img_file}: {exc}")

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_probs = np.array(all_probs)
    total = len(y_true)
    correct_mask = (y_true == y_pred)
    wrong_mask = ~correct_mask

    print(f"\nTotal test samples processed: {total}")
    print(f"Correct: {correct_mask.sum()}  |  Wrong: {wrong_mask.sum()}")
    print(f"Overall accuracy: {correct_mask.mean() * 100:.2f}%")

    # --- Confidence distributions ---
    conf_correct = np.max(y_probs[correct_mask], axis=1) * 100
    conf_wrong = np.max(y_probs[wrong_mask], axis=1) * 100 if wrong_mask.sum() > 0 else np.array([])

    print(f"\nConfidence statistics (correct predictions, n={len(conf_correct)}):")
    print(f"  Mean: {conf_correct.mean():.1f}%  Median: {np.median(conf_correct):.1f}%  "
          f"Min: {conf_correct.min():.1f}%  Max: {conf_correct.max():.1f}%")
    if len(conf_wrong) > 0:
        print(f"Confidence statistics (wrong predictions, n={len(conf_wrong)}):")
        print(f"  Mean: {conf_wrong.mean():.1f}%  Median: {np.median(conf_wrong):.1f}%  "
              f"Min: {conf_wrong.min():.1f}%  Max: {conf_wrong.max():.1f}%")

    # High-confidence errors (>=80%)
    print(f"\nHigh-confidence wrong predictions (confidence >= 80%): {len(error_samples)}")
    for e in error_samples:
        print(f"  GT: {e['gt_class']:20s} -> Pred: {e['pred_class']:20s} "
              f"({e['confidence_pct']:.1f}%)")

    # --- Per-class confidence ---
    print("\nPer-class confidence statistics:")
    per_class_conf = {}
    for idx, cls_name in enumerate(class_names):
        mask = (y_true == idx)
        if mask.sum() == 0:
            continue
        cls_confs = np.max(y_probs[mask], axis=1) * 100
        per_class_conf[cls_name] = {
            "n": int(mask.sum()),
            "correct": int((y_pred[mask] == idx).sum()),
            "mean_confidence": round(float(cls_confs.mean()), 2),
            "median_confidence": round(float(np.median(cls_confs)), 2),
            "min_confidence": round(float(cls_confs.min()), 2),
            "max_confidence": round(float(cls_confs.max()), 2)
        }
        print(f"  {cls_name:20s}: mean={cls_confs.mean():.1f}%  "
              f"min={cls_confs.min():.1f}%  max={cls_confs.max():.1f}%")

    # --- Calibration metrics ---
    ece, bin_bounds, confidences, accuracies = compute_ece(y_true, y_probs, n_bins=10)
    print(f"\nExpected Calibration Error (ECE, 10 bins): {ece:.4f}")

    # Brier Score (multi-class)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    brier = float(np.mean(np.sum((y_probs - y_bin) ** 2, axis=1)))
    print(f"Multi-class Brier Score: {brier:.4f}")

    # --- Reliability diagram ---
    fig, ax = plt.subplots(figsize=(7, 6))
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_counts = [], [], []

    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            bin_acc.append(accuracies[mask].mean())
            bin_conf.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_acc.append(0)
            bin_conf.append((lo + hi) / 2)
            bin_counts.append(0)

    bin_centers = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(n_bins)]
    bar_w = 1.0 / n_bins - 0.01

    ax.bar(bin_centers, bin_acc, width=bar_w, alpha=0.7, color="#198754",
           label="Observed Accuracy")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")
    ax.set_xlabel("Predicted Confidence", fontsize=12)
    ax.set_ylabel("Observed Accuracy", fontsize=12)
    ax.set_title(
        f"Reliability Diagram — PoultryScan AI 5-Class MobileNetV2\n"
        f"ECE = {ece:.4f}  |  Brier = {brier:.4f}  |  n = {total} test samples",
        fontsize=11
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    rel_path = os.path.join(output_dir, "reliability_diagram.png")
    fig.savefig(rel_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved reliability diagram: {rel_path}")

    # --- Confidence histogram ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if len(conf_correct) > 0:
        ax2.hist(conf_correct, bins=20, alpha=0.65, color="#198754",
                 label=f"Correct (n={len(conf_correct)})", density=True)
    if len(conf_wrong) > 0:
        ax2.hist(conf_wrong, bins=20, alpha=0.65, color="#dc3545",
                 label=f"Wrong (n={len(conf_wrong)})", density=True)
    ax2.set_xlabel("Softmax Confidence (%)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Confidence Distribution: Correct vs. Wrong Predictions", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    hist_path = os.path.join(output_dir, "confidence_histogram.png")
    fig2.savefig(hist_path, dpi=150)
    plt.close(fig2)
    print(f"Saved confidence histogram: {hist_path}")

    # --- Save report JSON ---
    report = {
        "model": model_path,
        "output_type": output_type,
        "output_sum_check": round(output_sum, 4),
        "caution": "High confidence does NOT guarantee correctness.",
        "total_test_samples": total,
        "correct": int(correct_mask.sum()),
        "wrong": int(wrong_mask.sum()),
        "overall_accuracy_pct": round(correct_mask.mean() * 100, 2),
        "confidence_correct": {
            "mean": round(float(conf_correct.mean()), 2),
            "median": round(float(np.median(conf_correct)), 2),
            "min": round(float(conf_correct.min()), 2),
            "max": round(float(conf_correct.max()), 2),
        } if len(conf_correct) > 0 else {},
        "confidence_wrong": {
            "mean": round(float(conf_wrong.mean()), 2),
            "median": round(float(np.median(conf_wrong)), 2),
            "min": round(float(conf_wrong.min()), 2),
            "max": round(float(conf_wrong.max()), 2),
        } if len(conf_wrong) > 0 else {},
        "per_class_confidence": per_class_conf,
        "high_confidence_errors_80pct_plus": len(error_samples),
        "high_confidence_error_samples": error_samples,
        "calibration": {
            "ece_10_bins": round(ece, 4),
            "brier_score_multiclass": round(brier, 4),
            "note": "ECE and Brier computed on test set. No thresholds adjusted from these results."
        }
    }
    report_path = os.path.join(output_dir, "confidence_error_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved error analysis report: {report_path}")
    print("=" * 68)
    return report


def main():
    parser = argparse.ArgumentParser(description="Confidence and error analysis for PoultryScan AI.")
    parser.add_argument("--model", default="model/poultry_disease_model.keras")
    parser.add_argument("--test-dir", default="test_dataset")
    parser.add_argument("--output", default="research_results")
    args = parser.parse_args()
    run_analysis(args.model, args.test_dir, args.output)


if __name__ == "__main__":
    main()
