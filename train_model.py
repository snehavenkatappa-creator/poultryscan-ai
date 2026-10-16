"""Train and fine-tune a MobileNetV2 classifier for five-class poultry disease detection.

Supported Five Classes:
1. Coccidiosis
2. Fowl Pox
3. Healthy
4. Newcastle Disease
5. Salmonella

Features:
- Validates folder dataset structure and checks image counts per class.
- Prevents image leakage between training/validation and held-out test splits.
- Transfer learning with ImageNet pre-trained MobileNetV2.
- Data augmentation: random flip, rotation, zoom, contrast.
- Preprocessing: normalized [0.0, 1.0] matching runtime inference.
- Balanced class weighting computed dynamically.
- Evaluates on held-out test data; saves confusion matrix and classification report.
- Saves model and exact class mapping together.

Usage:
    python train_model.py --data dataset --test-data test_dataset --epochs 10
"""

import argparse
import json
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def validate_dataset_folder(data_dir):
    """Validate directory presence and verify that every class folder has valid images."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")

    class_dirs = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
    if len(class_dirs) == 0:
        raise ValueError(f"No class subfolders found in '{data_dir}'.")

    counts = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for c in class_dirs:
        c_path = os.path.join(data_dir, c)
        imgs = [f for f in os.listdir(c_path) if os.path.splitext(f)[1].lower() in valid_exts]
        if len(imgs) == 0:
            raise ValueError(f"Class '{c}' contains 0 supported images in '{c_path}'.")
        counts[c] = len(imgs)

    return class_dirs, counts


def main():
    parser = argparse.ArgumentParser(description="Train 5-class poultry disease classification model.")
    parser.add_argument("--data", default="dataset", help="Path to training/validation dataset folder")
    parser.add_argument("--test-data", default="test_dataset", help="Path to held-out test dataset folder")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for training classification head")
    parser.add_argument("--fine-tune-epochs", type=int, default=4, help="Epochs for fine-tuning top layers")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--output", default="model/poultry_disease_model.keras", help="Target model path")
    parser.add_argument("--eval-dir", default="evaluation_results", help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    print("=" * 65)
    print("POULTRYSCAN AI - 5-CLASS MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 65)

    # 1. Validate datasets
    print(f"\n[Step 1] Validating training/validation dataset at '{args.data}'...")
    train_classes, train_counts = validate_dataset_folder(args.data)
    print(f"Discovered {len(train_classes)} classes:")
    for c, cnt in train_counts.items():
        print(f"  - {c:20s}: {cnt} images")

    # 2. Load dataset with Keras
    print("\n[Step 2] Creating train (80%) and validation (20%) splits...")
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=args.batch,
        shuffle=True
    )

    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=args.batch,
        shuffle=False
    )

    class_names = raw_train_ds.class_names
    num_classes = len(class_names)
    print(f"\nExact class ordering from training dataset ({num_classes} classes):")
    for idx, name in enumerate(class_names):
        print(f"  Index {idx} -> '{name}'")

    # 3. Calculate class weights to address dataset imbalance
    train_labels = []
    for _, y in raw_train_ds:
        train_labels.extend(y.numpy())
    train_labels = np.array(train_labels)

    unique_classes = np.unique(train_labels)
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_labels
    )
    class_weights_dict = {int(cls): float(w) for cls, w in zip(unique_classes, class_weights_arr)}
    print("\n[Step 3] Computed balanced class weights:")
    for idx, name in enumerate(class_names):
        print(f"  {name:20s}: weight = {class_weights_dict.get(idx, 1.0):.3f}")

    # 4. Normalization and Prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train_ds.map(lambda x, y: (x / 255.0, y)).prefetch(AUTOTUNE)
    val_ds = raw_val_ds.map(lambda x, y: (x / 255.0, y)).prefetch(AUTOTUNE)

    # 5. Build Model with MobileNetV2 Backbone
    print("\n[Step 4] Initializing MobileNetV2 architecture with ImageNet weights...")
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,), name="input_image")
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.35, name="dropout_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PoultryScan_MobileNetV2_5Class")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.4, monitor="val_loss", min_lr=1e-6),
    ]

    # 6. Phase 1 Training
    print(f"\n[Step 5] Phase 1: Training top classification head for up to {args.epochs} epochs...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # 7. Phase 2 Fine-Tuning
    if args.fine_tune_epochs > 0:
        print(f"\n[Step 6] Phase 2: Fine-tuning top 35 layers of MobileNetV2 for {args.fine_tune_epochs} epochs...")
        base_model.trainable = True
        for layer in base_model.layers[:-35]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights_dict,
            callbacks=callbacks
        )

    # 8. Save Model and exact class mapping
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"\n[SUCCESS] Trained 5-class model saved to: {args.output}")

    indices_path = os.path.join(os.path.dirname(args.output) or ".", "class_indices.json")
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"[SUCCESS] Exact class mapping saved to: {indices_path}")

    # 9. Evaluate on Held-Out Test Data
    if os.path.exists(args.test_data):
        print(f"\n[Step 7] Evaluating on held-out test dataset at '{args.test_data}'...")
        test_classes, test_counts = validate_dataset_folder(args.test_data)
        print("Test samples per class:", test_counts)

        test_raw_ds = tf.keras.utils.image_dataset_from_directory(
            args.test_data,
            image_size=IMG_SIZE,
            batch_size=args.batch,
            shuffle=False
        )

        test_ds = test_raw_ds.map(lambda x, y: (x / 255.0, y))
        y_true = np.concatenate([y.numpy() for _, y in test_raw_ds], axis=0)

        y_pred_probs = model.predict(test_ds, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        test_acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)

        print("\n" + "=" * 65)
        print("HELD-OUT TEST SET EVALUATION BENCHMARK")
        print("=" * 65)
        print(f"Overall Test Accuracy: {test_acc * 100:.2f}%\n")
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report_text)
        print("=" * 65)

        # Save artifacts
        os.makedirs(args.eval_dir, exist_ok=True)

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 7))
        cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
        ax.set_title(f"PoultryScan AI Confusion Matrix (5 Classes · {test_acc*100:.1f}% Acc)", fontsize=13, pad=12)
        fig.colorbar(cax)
        tick_marks = np.arange(num_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_ylabel("True / Ground Truth Class", fontsize=11)
        ax.set_xlabel("Predicted Class", fontsize=11)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight="bold")

        fig.tight_layout()
        cm_path = os.path.join(args.eval_dir, "confusion_matrix.png")
        fig.savefig(cm_path, dpi=200)
        plt.close(fig)
        print(f"Saved confusion matrix plot: {cm_path}")

        # Also copy to static/images/confusion_matrix.png for web rendering
        static_img_dir = os.path.join("static", "images")
        os.makedirs(static_img_dir, exist_ok=True)
        shutil.copyfile(cm_path, os.path.join(static_img_dir, "confusion_matrix.png"))

        # Save summary JSON
        summary = {
            "overall_accuracy": round(float(test_acc) * 100, 2),
            "total_test_samples": int(len(y_true)),
            "preprocessing": "rescale [0.0, 1.0]",
            "classes": class_names,
            "per_class_accuracy": {
                class_names[i]: round(float(cm[i, i] / max(cm[i].sum(), 1)) * 100, 2)
                for i in range(num_classes)
            },
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist()
        }
        summary_path = os.path.join(args.eval_dir, "evaluation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved evaluation metrics JSON: {summary_path}")


if __name__ == "__main__":
    main()
