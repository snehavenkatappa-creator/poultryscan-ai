"""One-shot model inspection script."""
import os, json, zipfile
import numpy as np

# --- 1. Read model config WITHOUT loading TF (fast) ---
model_path = "model/poultry_disease_model.keras"
print("=== .keras Archive Inspection ===")
with zipfile.ZipFile(model_path, "r") as zf:
    metadata = json.loads(zf.read("metadata.json"))
    config = json.loads(zf.read("config.json"))

print(f"Saved with Keras: {metadata.get('keras_version')}")
print(f"Date saved:       {metadata.get('date_saved')}")

layers = config["config"]["layers"]
last_dense = None
for layer in layers:
    if layer.get("class_name") == "Dense":
        last_dense = layer
print(f"Output Dense layer: units={last_dense['config']['units']}, activation={last_dense['config']['activation']}")
print()

# --- 2. Load with TF and run a dummy prediction ---
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

model = tf.keras.models.load_model(model_path, compile=False)
print(f"Model input shape:  {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print(f"Number of output classes: {model.output_shape[-1]}")

dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
preds = model.predict(dummy, verbose=0)
print(f"Dummy prediction vector: {preds[0]}")
print(f"Predicted index: {np.argmax(preds[0])}")
print()

# --- 3. Check class_indices.json ---
ci_path = "model/class_indices.json"
if os.path.exists(ci_path):
    with open(ci_path) as f:
        class_indices = json.load(f)
    print(f"class_indices.json: {class_indices}")
else:
    print("class_indices.json: NOT FOUND (will be created during retraining)")

# --- 4. Confirm current diseases.py CLASS_NAMES ---
print()
print("=== diseases.py CLASS_NAMES ===")
import sys
sys.path.insert(0, ".")
from diseases import CLASS_NAMES, ACTIVE_MODEL_CLASSES, REFERENCE_CLASSES
print(f"CLASS_NAMES ({len(CLASS_NAMES)}): {CLASS_NAMES}")
print(f"ACTIVE_MODEL_CLASSES: {ACTIVE_MODEL_CLASSES}")
print(f"REFERENCE_CLASSES: {REFERENCE_CLASSES[:5]}...")

print()
print("=== DIAGNOSIS SUMMARY ===")
print(f"Model output neurons: {model.output_shape[-1]}")
print(f"diseases.py CLASS_NAMES count: {len(CLASS_NAMES)}")
if model.output_shape[-1] == 4 and "Fowl Pox" not in CLASS_NAMES:
    print("STATUS: Model is a 4-class model (Coccidiosis, Healthy, NCD, Salmonella). Fowl Pox is NOT in CLASS_NAMES.")
    print("ROOT CAUSE: The model was never retrained to include Fowl Pox.")
elif model.output_shape[-1] == 5 and "Fowl Pox" in CLASS_NAMES:
    print("STATUS: Model has 5 outputs and Fowl Pox is in CLASS_NAMES - check class ordering!")
else:
    print(f"STATUS: Mismatch - model outputs={model.output_shape[-1]}, CLASS_NAMES={CLASS_NAMES}")
