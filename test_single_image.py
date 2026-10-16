"""Test the trained poultry disease model on a single image.

Usage:
    python test_single_image.py --image "path/to/image.jpg"
    python test_single_image.py --image "path/to/image.jpg" --preprocess rescale
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

import json

# Fallback class order if class_indices.json is unavailable
DEFAULT_CLASS_NAMES = ["Coccidiosis", "Fowl Pox", "Healthy", "Newcastle Disease", "Salmonella"]
IMG_SIZE = (224, 224)


def load_class_names(indices_path="model/class_indices.json"):
    """Load exact class mapping JSON next to the model."""
    if os.path.exists(indices_path):
        try:
            with open(indices_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data[str(i)] for i in range(len(data))]
        except Exception:
            pass
    return DEFAULT_CLASS_NAMES


def load_poultry_model(model_path="model/poultry_disease_model.keras"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    import keras
    return keras.models.load_model(model_path, compile=False)


def preprocess_image(image_path, method="rescale"):
    """Preprocess single image for MobileNetV2 input.
    
    Args:
        image_path: Path to the image file.
        method: 'rescale' ([0, 1] range) or 'mobilenet' ([-1, 1] range).
    """
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32")
    if method == "rescale":
        arr = arr / 255.0
    else:
        arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def predict_image(model, image_path, class_names=None, method="rescale"):
    """Run inference on an image and return class, confidence, and class probabilities."""
    if class_names is None:
        class_names = load_class_names()
    x = preprocess_image(image_path, method=method)
    probs = model.predict(x, verbose=0)[0]  # Softmax probabilities
    top_idx = int(np.argmax(probs))
    top_label = class_names[top_idx] if top_idx < len(class_names) else f"Class_{top_idx}"
    top_conf = float(probs[top_idx]) * 100

    all_probs = {
        class_names[i]: round(float(probs[i]) * 100, 2)
        for i in range(min(len(class_names), len(probs)))
    }
    return top_label, top_conf, all_probs


def main():
    parser = argparse.ArgumentParser(description="Predict poultry disease from an image.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="model/poultry_disease_model.keras", help="Path to .keras model")
    parser.add_argument("--indices", default="model/class_indices.json", help="Path to class_indices.json")
    parser.add_argument("--preprocess", choices=["rescale", "mobilenet"], default="rescale",
                        help="Normalization scheme (rescale=[0, 1] verified test accuracy: 91.32% on 242 samples)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_poultry_model(args.model)

    class_names = load_class_names(args.indices)
    print(f"Testing image: {args.image}")
    label, confidence, probs = predict_image(model, args.image, class_names=class_names, method=args.preprocess)

    print("\n" + "=" * 50)
    print(f"PREDICTION RESULT")
    print("=" * 50)
    print(f"Predicted Disease: {label}")
    print(f"Confidence:        {confidence:.2f}%")
    print("-" * 50)
    print("Class Probabilities:")
    for c, p in probs.items():
        bar = "#" * int(p // 5)
        print(f"  {c:20s}: {p:6.2f}% | {bar}")
    print("=" * 50)


if __name__ == "__main__":
    main()
