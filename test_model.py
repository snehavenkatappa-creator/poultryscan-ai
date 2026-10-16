"""CLI Test & Inference Script for PoultryScan AI 5-Class Model.

Usage:
    python test_model.py --image "path/to/chicken_sample.jpg"
    python test_model.py --image "path/to/chicken_sample.jpg" --mode head_face
    python test_model.py --dir "path/to/folder_of_images"

Outputs:
    - Predicted disease class
    - Confidence percentage
    - Class probability breakdown across all 5 classes
    - Uncertainty flag and veterinary advisory note
"""

import argparse
import json
import os
import sys
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 65.0

def load_class_mapping(indices_path="model/class_indices.json"):
    """Load class indices list/dict from json file."""
    if os.path.exists(indices_path):
        with open(indices_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Sort by integer keys or values
                try:
                    return [data[str(i)] for i in range(len(data))]
                except KeyError:
                    return list(data.keys())
    # Fallback to default 5 classes in alphabetical order
    return ["Coccidiosis", "Fowl Pox", "Healthy", "Newcastle Disease", "Salmonella"]


def load_model(model_path="model/poultry_disease_model.keras"):
    """Load the trained Keras model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please train the model first.")
    
    try:
        import keras
        return keras.models.load_model(model_path, compile=False)
    except Exception:
        from tensorflow.keras.models import load_model as tf_load
        return tf_load(model_path, compile=False)


def preprocess_image(image_path):
    """Normalize input image to [0.0, 1.0] matching training configuration."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, class_names, image_path, mode="full_body"):
    """Run inference on a single image."""
    tensor = preprocess_image(image_path)
    probs = model.predict(tensor, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_class = class_names[top_idx]
    top_conf = float(probs[top_idx]) * 100.0

    sorted_probs = np.sort(probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
    num_classes = len(probs)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(num_classes))

    is_uncertain = (top_conf < CONFIDENCE_THRESHOLD) or (margin < 0.20) or (entropy > 0.65)

    prob_dict = {
        class_names[i]: round(float(probs[i]) * 100.0, 2)
        for i in range(len(class_names))
    }

    return {
        "file": os.path.basename(image_path),
        "predicted_class": top_class,
        "confidence_pct": round(top_conf, 2),
        "is_uncertain": is_uncertain,
        "probabilities": prob_dict,
        "margin_pct": round(margin * 100.0, 2),
        "entropy": round(entropy, 3)
    }


def main():
    parser = argparse.ArgumentParser(description="Test PoultryScan AI model on sample images.")
    parser.add_argument("--image", help="Path to single image file")
    parser.add_argument("--dir", help="Path to directory of image files")
    parser.add_argument("--model", default="model/poultry_disease_model.keras", help="Path to .keras model")
    parser.add_argument("--indices", default="model/class_indices.json", help="Path to class_indices.json")
    parser.add_argument("--mode", default="full_body", choices=["full_body", "head_face"], help="Examination mode")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        sys.exit(1)

    print("Loading class mapping and model...")
    class_names = load_class_mapping(args.indices)
    model = load_model(args.model)
    print(f"Loaded model with {model.output_shape[-1]} outputs. Classes: {class_names}\n")

    files_to_test = []
    if args.image:
        files_to_test.append(args.image)
    if args.dir:
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        for f in sorted(os.listdir(args.dir)):
            if os.path.splitext(f)[1].lower() in valid_exts:
                files_to_test.append(os.path.join(args.dir, f))

    for img_path in files_to_test:
        res = predict(model, class_names, img_path, mode=args.mode)
        print("=" * 60)
        print(f"Sample: {res['file']}")
        print(f"Predicted Diagnosis: {res['predicted_class']} ({res['confidence_pct']}%)")
        print(f"Status: {'[LOW CONFIDENCE / UNCERTAIN]' if res['is_uncertain'] else '[HIGH CONFIDENCE]'}")
        print("Probability Distribution:")
        for cname, p in res["probabilities"].items():
            bar = "#" * int(p / 5)
            print(f"  {cname:20s}: {p:6.2f}% | {bar}")
        print("=" * 60)


if __name__ == "__main__":
    main()
