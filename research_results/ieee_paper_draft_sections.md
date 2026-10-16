# PoultryScan AI — IEEE Paper Draft Sections
# Experimental Setup, Evaluation Metrics, Results, and Limitations

> **STATUS NOTE:** All metrics in this draft reflect MEASURED experimental results
> from scripts run on 2026-10-14. No values are fabricated or estimated.
> Sections marked [PENDING] require additional data or future experiments.

---

## III. EXPERIMENTAL SETUP

### A. Dataset

The dataset comprises **2,057 images** distributed across five disease classes:
Coccidiosis, Fowl Pox, Healthy, Newcastle Disease, and Salmonella. Images were
sourced from [**DATASET SOURCE NOT DOCUMENTED** — must be disclosed before submission;
cite original source and license].

| Class             | Training | Test | Total | Imbalance vs. Majority |
|-------------------|----------|------|-------|------------------------|
| Coccidiosis       | 450      | 56   | 506   | 1.00×                  |
| Fowl Pox          | 127      | 28   | 155   | 3.54×                  |
| Healthy           | 450      | 60   | 510   | 1.00×                  |
| Newcastle Disease | 338      | 38   | 376   | 1.33×                  |
| Salmonella        | 450      | 60   | 510   | 1.00×                  |
| **Total**         | **1,815**| **242** | **2,057** |               |

The training set was split 80/20 into effective training (~1,452 images) and
validation (~363 images) subsets using `tf.keras.utils.image_dataset_from_directory`
with `validation_split=0.2` and `seed=42`. The held-out test set (242 images) was
stored in a separate directory and was not used during any phase of training,
validation-based model selection, or threshold tuning. SHA-256 hashing of all
1,785 training images and 242 test images confirmed zero byte-level duplicates
across the two splits.

**Note on near-duplicate detection:** The dataset contains no metadata identifying
bird identity, farm of origin, or video sequence. Near-duplicate detection
(same bird, different angle or crop) was therefore not feasible. This is a
known limitation that could introduce correlated samples within and across splits.

**Note on Fowl Pox imbalance:** Fowl Pox has 127 training images compared to 450
for three other classes. Balanced class weights were computed via
`sklearn.utils.class_weight.compute_class_weight('balanced')` and applied during
training to mitigate this imbalance.

### B. Model Architecture

We adopt **MobileNetV2** [Howard et al., 2018] pre-trained on ImageNet as a
feature extractor. The classification head is:

```
Input (224×224×3)
→ Data Augmentation (RandomFlip, RandomRotation±0.12, RandomZoom±0.12, RandomContrast±0.1)
→ MobileNetV2 backbone (input_shape=224×224×3, include_top=False, weights='imagenet')
→ GlobalAveragePooling2D
→ Dropout (rate=0.35)
→ Dense (5, activation='softmax')
```

**Training Protocol:**
- Phase 1: MobileNetV2 frozen; top classification head trained for up to 10 epochs
  (Adam, lr=1e-3, EarlyStopping patience=4, ReduceLROnPlateau patience=2, factor=0.4)
- Phase 2 (fine-tuning): Top 35 layers of MobileNetV2 unfrozen; trained for up to 4 epochs
  (Adam, lr=1e-5, same callbacks)
- Loss: sparse categorical cross-entropy with balanced class weights
- Batch size: 32 | Image size: 224×224 | Preprocessing: pixel values ÷ 255.0 → [0, 1]

The model was saved as `model/poultry_disease_model.keras` (Keras 3.14.0, 22.7 MB).
Both the Head/Face Analysis and Full Body/Droppings Analysis examination modes load
the same single model. No separate mode-specific models were trained.

### C. Evaluation Protocol

Evaluation was performed on the 242-image held-out test set using `evaluate_model.py`.
Preprocessing at inference exactly matches training: RGB conversion, bilinear resize to
224×224, and division by 255.0.

Class order used: `['Coccidiosis', 'Fowl Pox', 'Healthy', 'Newcastle Disease', 'Salmonella']`
(alphabetical folder order, stored in `model/class_indices.json`).

---

## IV. EVALUATION METRICS

We report the following metrics as recommended for multi-class imbalanced classification:

- **Overall accuracy** — proportion of correct predictions over all 242 test samples
- **Per-class precision, recall, and F1-score** — using scikit-learn's `classification_report`
  with `zero_division=0`
- **Macro-average precision, recall, F1** — unweighted mean across all 5 classes
  (treats all classes equally regardless of support)
- **Weighted-average precision, recall, F1** — weighted by per-class support
- **Confusion matrix** — rows = ground truth, columns = predicted, with labeled axes
- **Per-class support** — number of test images per class

For calibration analysis (Phase 4):
- **Expected Calibration Error (ECE)** — 10-bin equal-width, measures alignment between
  predicted confidence and observed accuracy
- **Multi-class Brier Score** — mean squared error between one-hot labels and softmax vectors
- **Reliability diagram** — visual calibration plot

All metrics were computed on the test set only. No thresholds or decision boundaries
were tuned using test data.

---

## V. RESULTS

### A. Classification Performance

All metrics are from a single evaluation run on 2026-10-14 (242 held-out test images,
zero SHA-256 leakage confirmed).

**Overall test accuracy: 91.32%** (221/242 correct)

> *Note:* The value 94.44% appearing in `diseases.py` line 4 is a stale comment
> referring to an earlier 4-class model (saved 2026-08-03) that did not include
> Fowl Pox and is no longer loaded by the application. The verified result for
> the current 5-class model is **91.32%**.

#### Per-Class Metrics (5-Class MobileNetV2, 242 Test Samples)

| Class             | Precision | Recall  | F1-Score | Support |
|-------------------|-----------|---------|----------|---------|
| Coccidiosis       | 1.0000    | 0.9464  | 0.9725   | 56      |
| Fowl Pox          | 1.0000    | 1.0000  | 1.0000   | 28      |
| Healthy           | 0.8485    | 0.9333  | 0.8889   | 60      |
| Newcastle Disease | 0.8293    | 0.8947  | 0.8608   | 38      |
| Salmonella        | 0.9259    | 0.8333  | 0.8772   | 60      |
| **Macro avg**     | **0.9207**| **0.9216**| **0.9199**| 242  |
| **Weighted avg**  | **0.9173**| **0.9132**| **0.9138**| 242  |
| **Accuracy**      |           |         | **0.9132**| 242    |

Source: `evaluation_results/evaluation_summary.json` (generated by `evaluate_model.py`)

#### Confusion Matrix

```
                  Predicted →
Actual ↓   Cocci  FPox  Healthy  NCD  Salmo
Coccidiosis   53     0      1      2      0
Fowl Pox       0    28      0      0      0
Healthy        0     0     56      2      2
Newcastle D.   0     0      2     34      2
Salmonella     0     0      7      3     50
```

The most frequent misclassifications:
1. Salmonella → Healthy (7 samples) — likely due to visually similar fecal appearance
2. Salmonella → Newcastle Disease (3 samples)
3. Healthy → Newcastle Disease (2 samples)
4. Newcastle Disease → Healthy (2 samples)

**Fowl Pox caveat:** Perfect recall (28/28) on only 28 test samples is insufficient to
claim robust generalization. A larger, more diverse Fowl Pox test set is needed.

### B. Confidence and Calibration Analysis

Output type confirmed: **softmax probabilities** (output sums to ~1.0 per sample).

| Metric                                | Value    |
|---------------------------------------|----------|
| Mean confidence (correct, n=216)      | 91.6%    |
| Median confidence (correct, n=216)    | 97.5%    |
| Mean confidence (wrong, n=26)         | 60.8%    |
| Median confidence (wrong, n=26)       | 56.4%    |
| Max confidence (wrong prediction)     | 94.4%    |
| High-confidence wrong (≥80%), n=5     | 5        |
| Expected Calibration Error (ECE)      | 0.0410   |
| Multi-class Brier Score               | 0.1428   |

**Key finding:** The model has a low ECE of 0.041, indicating generally well-calibrated
confidence. However, 5 incorrect predictions occurred with ≥80% confidence — all
involving Salmonella↔Healthy confusion. High confidence does NOT guarantee correctness.

Per-class mean confidence:

| Class             | Mean Conf. | Min   | Max    |
|-------------------|------------|-------|--------|
| Coccidiosis       | 93.1%      | 49.4% | 100.0% |
| Fowl Pox          | 97.8%      | 74.8% | 100.0% |
| Healthy           | 86.6%      | 40.8% | 99.9%  |
| Newcastle Disease | 79.1%      | 29.9% | 99.7%  |
| Salmonella        | 87.0%      | 50.1% | 100.0% |

Newcastle Disease has the lowest mean confidence, consistent with its highest
confusion rate.

### C. Grad-CAM Qualitative Analysis

All visualizations are stored in `research_results/grad_cam_examples/` with metadata recorded in `gradcam_manifest.json`.

Grad-CAM targets the `out_relu` layer (the terminal 4-D spatial activation map of MobileNetV2, shape 7×7×1280, immediately preceding GlobalAveragePooling2D). Each figure displays a 3-panel layout: original input specimen, raw jet colormap gradient heatmap, and blended alpha overlay (α=0.45).

**Empirical qualitative observations across representative samples:**
- **Coccidiosis (Fecal):** Saliency maps (e.g., `cocci.1176.jpg`, `cocci.1539.jpg`) reliably concentrate on pathognomonic reddish/dark hemorrhagic mucosal streaks and high-fluidity fecal deposits rather than peripheral bedding or background.
- **Fowl Pox (Cutaneous Head/Face):** Visualizations (e.g., `fowlpox_100.jpg`, `fowlpox_81.jpg`) exhibit focused gradient activation localized directly over discrete nodular crusts and wart-like papules on comb and facial skin margins.
- **Healthy Specimens:** Saliency centers on compact, well-formed fecal boluses with distinct white urate caps (`healthy.1343.jpg`, `healthy.1383.jpg`), confirming attention to normal digestive markers.
- **Newcastle Disease:** The network concentrates attention on profuse watery emerald-green liquid diarrhea (`ncd.153.jpg`).
- **Error Analysis Cases:**
  - `ncd.340.jpg` (Ground truth: Newcastle Disease; Predicted: Fowl Pox, 53.0% confidence): Ambient lighting reflections on comb tissue produced localized high-frequency texture resembling pox-like papules, illustrating a visual shortcut susceptibility under suboptimal farm illumination.
  - `salmo.1335.jpg` (Ground truth: Salmonella; Predicted: Healthy, 55.5% confidence): Fecal matter lacked distinct chalky-white urate encrustation, leading to high feature overlap with normal fecal morphology.
- **Scientific Limitation:** Grad-CAM illustrates gradient sensitivity of the predicted class score with respect to convolutional feature activations. It serves as an explanatory diagnostic aid, not proof of clinical causality.

---

## VI. LIMITATIONS

1. **Single model for dual modes.** Both examination modes (Head/Face and Full Body/
   Droppings) share a single MobileNetV2 model trained on mixed image types. No
   mode-specific models exist. The clinical value of mode separation is therefore
   limited to user guidance, not different inference.

2. **Fowl Pox class imbalance.** With only 127 training images (vs. 450 for majority
   classes), Fowl Pox metrics (especially the 100% test recall) may not generalize.

3. **Unknown dataset provenance.** The source, geographic origin, camera hardware,
   flock breed, bird age, and licensing for all images are undocumented in this
   codebase. This must be disclosed in the final paper.

4. **Near-duplicate risk.** SHA-256 hashing confirmed zero byte-level duplicates.
   However, images from the same bird, pen, or video sequence could remain across
   splits and inflate test performance. This cannot be resolved without metadata.

5. **No generalization evaluation.** The model was trained and tested on a single
   dataset. Generalization to different farms, breeds, geographic regions, camera
   types, or lighting conditions has not been evaluated.

6. **Preprocessing mismatch note.** The model uses custom rescaling (/255.0) rather
   than MobileNetV2's canonical `preprocess_input` (which maps to [-1, 1]). Training
   and inference are consistent with each other, but this differs from the ImageNet
   pretraining distribution. Potential impact on feature extraction has not been
   ablated.

7. **Screening tool only.** PoultryScan AI is a preliminary triage tool. Results
   should not replace veterinary laboratory diagnostics (necropsy, culture, RT-PCR).

8. **4-class backup model.** The `poultry_disease_model_4class_backup.keras` (saved
   2026-08-03, 4-output Dense) is retired and has no paired class_indices.json.
   Its class mapping is unknown. It is NOT used by the application and MUST NOT
   be cited in performance claims.

---

## VII. UNRESOLVED ISSUES AND DATA NEEDED

| # | Issue | Data/Action Needed |
|---|-------|--------------------|
| 1 | Dataset source, license, provenance | Document all image sources before submission |
| 2 | Fowl Pox small test set (28 images) | Collect ≥100 additional Fowl Pox test images |
| 3 | No mode-specific models | Either train separate head/face vs. fecal models, or explicitly document shared model in paper |
| 4 | Backup 4-class model class mapping | Locate original training script/log or retire permanently |
| 5 | Near-duplicate detection | Obtain metadata (bird ID, video source) for proper leakage audit |
| 6 | Grad-CAM visual quality review | Manual inspection of generated figures required |
| 7 | Comparison baselines | Identify and cite comparable prior work (after verification) |
| 8 | Cross-dataset generalization | Evaluate on an external dataset from a different source |
| 9 | Hardware environment documentation | Record GPU/CPU used during training for reproducibility |
| 10 | EfficientNet model in zip | Unzip, inspect architecture, class mapping, and compare to MobileNetV2 |
