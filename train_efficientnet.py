"""Train EfficientNetB0 for 5-class poultry disease detection.

Exact mirror of MobileNetV2 training protocol for fair algorithm comparison.
"""

import json
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "dataset"
TEST_DIR = "test_dataset"
OUTPUT_PATH = "model/efficientnet_poultry_trained.keras"
EVAL_DIR = "evaluation_results"
CLASS_INDICES_OUTPUT = "model/efficientnet_class_indices.json"
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 4

print("=" * 68)
print("POULTRYSCAN AI - EFFICIENTNETB0 TRAINING")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 68)

raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

class_names = raw_train_ds.class_names
num_classes = len(class_names)
print(f"Classes: {class_names}")

train_labels = np.concatenate([y.numpy() for _, y in raw_train_ds])
unique_classes = np.unique(train_labels)
class_weights_arr = compute_class_weight("balanced", classes=unique_classes, y=train_labels)
class_weights_dict = {int(cls): float(w) for cls, w in zip(unique_classes, class_weights_arr)}
print("Class weights:", class_weights_dict)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = raw_train_ds.map(lambda x, y: (x / 255.0, y)).prefetch(AUTOTUNE)
val_ds = raw_val_ds.map(lambda x, y: (x / 255.0, y)).prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"), layers.RandomRotation(0.12),
    layers.RandomZoom(0.12), layers.RandomContrast(0.1),
], name="data_augmentation")

base_model = EfficientNetB0(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,), name="input_image")
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = layers.Dropout(0.35, name="dropout_head")(x)
outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)
model = models.Model(inputs=inputs, outputs=outputs, name="PoultryScan_EfficientNetB0")

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss", verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.4, monitor="val_loss", min_lr=1e-6, verbose=1),
]

print("\nPhase 1: Training head...")
h1 = model.fit(train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS, class_weight=class_weights_dict, callbacks=callbacks, verbose=1)
best1 = max(h1.history.get("val_accuracy", [0]))
print(f"Phase 1 best val accuracy: {best1*100:.2f}%")

print("\nPhase 2: Fine-tuning top 20 layers...")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h2 = model.fit(train_ds, validation_data=val_ds, epochs=PHASE2_EPOCHS, class_weight=class_weights_dict, callbacks=callbacks, verbose=1)
best2 = max(h2.history.get("val_accuracy", [0]))
print(f"Phase 2 best val accuracy: {best2*100:.2f}%")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
model.save(OUTPUT_PATH)
print(f"\nModel saved: {OUTPUT_PATH}")
with open(CLASS_INDICES_OUTPUT, "w") as f:
    json.dump(class_names, f, indent=2)

print("\nEvaluating on test set...")
test_raw_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
test_ds = test_raw_ds.map(lambda x, y: (x / 255.0, y))
y_true = np.concatenate([y.numpy() for _, y in test_raw_ds])
y_pred_probs = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
print(f"\nOverall Test Accuracy: {acc*100:.2f}%")
print(report_text)

os.makedirs(EVAL_DIR, exist_ok=True)
per_class_recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
summary = {
    "model": "EfficientNetB0", "model_file": OUTPUT_PATH,
    "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "test_set_results": {
        "overall_accuracy": round(float(acc)*100, 2),
        "total_test_samples": int(len(y_true)),
        "macro_f1": round(report_dict["macro avg"]["f1-score"]*100, 2),
        "macro_precision": round(report_dict["macro avg"]["precision"]*100, 2),
        "macro_recall": round(report_dict["macro avg"]["recall"]*100, 2),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"]*100, 2),
        "per_class_recall": {class_names[i]: round(float(per_class_recalls[i])*100, 2) for i in range(num_classes)},
        "confusion_matrix": cm.tolist(), "classification_report": report_dict
    },
    "classes": class_names,
    "phase1_best_val_accuracy": round(float(best1)*100, 2),
    "phase2_best_val_accuracy": round(float(best2)*100, 2),
}
summary_path = os.path.join(EVAL_DIR, "efficientnet_evaluation_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
shutil.copy(summary_path, os.path.join("research_results", "efficientnet_evaluation_summary.json"))
print(f"\nSaved: {summary_path}")
print("\nDONE")
