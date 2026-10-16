# Reproducibility README — PoultryScan AI Research Evaluation

## Overview

This directory contains all evaluation scripts, metrics, figures, and draft paper sections
for the PoultryScan AI IEEE conference submission.

**Model:** MobileNetV2-based 5-class poultry disease classifier  
**Framework:** TensorFlow 2.16.1 / Keras 3.14.0  
**Evaluation Date:** 2026-10-14  

---

## Directory Structure

```
research_results/
├── README_reproducibility.md       ← This file
├── model_inventory.json            ← All model files, architectures, class mappings
├── dataset_split_summary.json      ← Dataset counts, imbalance, leakage check
├── per_class_metrics.csv           ← Classification report in CSV format
├── confusion_matrix_5class.png     ← High-resolution confusion matrix (200 DPI)
├── confidence_error_report.json    ← Calibration, ECE, Brier, error analysis
├── reliability_diagram.png         ← Calibration reliability diagram
├── confidence_histogram.png        ← Confidence distribution: correct vs wrong
├── grad_cam_examples/
│   ├── gradcam_manifest.json       ← Metadata for all Grad-CAM figures
│   └── *.png                       ← 3-panel Grad-CAM figures per test image
└── eval_scripts/
    ├── grad_cam_analysis.py        ← Grad-CAM generation script
    ├── confidence_error_analysis.py← Calibration & error analysis script
    └── dataset_split_summary.py    ← Dataset audit script
```

The main evaluation scripts in the project root are:
- `evaluate_model.py` — full per-class metrics, confusion matrix, leakage check
- `train_model.py`    — training pipeline (do not rerun without new data)
- `inspect_model.py`  — model architecture inspection (no TF needed for step 1)

---

## Reproducing the Evaluation

### Requirements

```
tensorflow==2.16.1
keras==3.14.0
numpy
scikit-learn
matplotlib
Pillow
flask
```

Install:
```bash
pip install -r requirements.txt
```

### Step 1 — Inspect model (no dataset needed)

```bash
python inspect_model.py
```

Expected output:
- Keras version: 3.14.0, saved 2026-10-14
- Output Dense: units=5, activation=softmax
- class_indices.json: ['Coccidiosis', 'Fowl Pox', 'Healthy', 'Newcastle Disease', 'Salmonella']

### Step 2 — Full evaluation on held-out test set

```bash
python evaluate_model.py \
    --model model/poultry_disease_model.keras \
    --data test_dataset \
    --train-data dataset \
    --preprocess rescale \
    --output evaluation_results
```

Expected key results (already computed, 2026-10-14 15:07:29):
- Overall accuracy: **91.32%** on **242 test images**
- Macro F1: 0.9199
- Zero SHA-256 duplicates between dataset/ and test_dataset/

### Step 3 — Confidence & calibration analysis

```bash
python research_results/eval_scripts/confidence_error_analysis.py \
    --model model/poultry_disease_model.keras \
    --test-dir test_dataset \
    --output research_results
```

### Step 4 — Grad-CAM qualitative visualization

```bash
python research_results/eval_scripts/grad_cam_analysis.py \
    --model model/poultry_disease_model.keras \
    --test-dir test_dataset \
    --output research_results/grad_cam_examples \
    --samples-per-class 3 \
    --seed 42
```

### Step 5 — Dataset audit

```bash
python research_results/eval_scripts/dataset_split_summary.py
```

---

## Random Seeds

| Script                       | Seed | Purpose                    |
|------------------------------|------|----------------------------|
| train_model.py               | 42   | train/val split            |
| grad_cam_analysis.py         | 42   | image sampling per class   |
| evaluate_model.py            | N/A  | shuffle=False, deterministic|

---

## Software Environment

| Package      | Version  |
|--------------|----------|
| Python       | 3.11/3.12|
| TensorFlow   | 2.16.1   |
| Keras        | 3.14.0   |
| NumPy        | (pinned in requirements.txt) |
| scikit-learn | (pinned) |
| Pillow       | (pinned) |
| Flask        | (pinned) |

Hardware: CPU-only (oneDNN custom operations enabled)

---

## Critical Notes for IEEE Paper

1. **Single model for both modes.** Despite the website describing two examination modes
   (Head/Face and Full Body/Droppings), only ONE model is loaded for both.
   The 4-class backup (`poultry_disease_model_4class_backup.keras`, saved 2026-08-03)
   is NOT loaded by the application and has an unknown class mapping.

2. **Accuracy discrepancy resolved.**
   - The website's `evaluation_summary.json` (run 2026-10-14 15:07:29) reports **91.32%**
     on 242 test images — this is the confirmed, measured result.
   - `diseases.py` line 4 contains a hardcoded claim of **94.44%** referring to a
     "4 Active Model Classes" which is outdated (pre-Fowl Pox retraining). This claim
     is STALE and INCORRECT for the current 5-class model.
   - The figure **91.32%** is the verified result for the current deployed model.

3. **Preprocessing.** Training and inference both use `/255.0` rescaling to [0, 1].
   The MobileNetV2 `preprocess_input` function (which maps to [-1, 1]) is NOT used.
   This is intentional and consistent between train and inference.

4. **Fowl Pox class imbalance.** Only 127 training images vs 450 for Coccidiosis,
   Healthy, and Salmonella. Balanced class weights were applied during training.
   The model achieves 100% recall on Fowl Pox in the test set (28/28), but the
   small test support makes this statistic uncertain.

5. **Data leakage.** SHA-256 duplicate check on 1,785 training images vs 242 test
   images: **zero duplicates found**. Near-duplicate detection (same bird, different
   angle) is not possible without metadata.

6. **Grad-CAM target layer.** The last 4-D spatial feature map inside the MobileNetV2
   sub-model is used. The exact layer name is logged in `gradcam_manifest.json`.
   Grad-CAM is a qualitative tool only.
