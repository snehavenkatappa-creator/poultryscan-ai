"""PoultryScan AI — Chicken Presence & Multi-Chicken Flock Detection Module.

Uses a lightweight, pre-trained FasterRCNN-MobileNetV3 detector (COCO 91 classes)
to locate chickens/birds, distinguish unrelated entities (people, cars, animals, objects),
and perform multi-chicken flock segmentation and individual crop analysis.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("PoultryScanAI.ChickenDetector")

_detector_model = None
_detector_transforms = None
_detector_categories = None


def get_detector():
    """Lazy load FasterRCNN-MobileNetV3 detector once and cache in memory."""
    global _detector_model, _detector_transforms, _detector_categories
    if _detector_model is not None:
        return _detector_model, _detector_transforms, _detector_categories

    try:
        import torch
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        )

        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        model.eval()
        transforms = weights.transforms()
        categories = weights.meta["categories"]

        _detector_model = model
        _detector_transforms = transforms
        _detector_categories = categories
        logger.info("FasterRCNN-MobileNetV3 chicken detector loaded successfully.")
        return _detector_model, _detector_transforms, _detector_categories
    except Exception as exc:
        logger.error("Failed to load FasterRCNN detector: %s", exc)
        return None, None, None


def detect_specimens(image_path_or_pil, bird_thresh=0.35, non_bird_thresh=0.35):
    """Run object detection on an image.
    
    Returns:
        birds: list of dicts {"id": int, "box": [x1, y1, x2, y2], "score": float, "bw": float, "bh": float}
        non_birds: list of dicts {"label": str, "score": float, "box": [x1, y1, x2, y2]}
    """
    model, transforms, categories = get_detector()
    if model is None:
        return [], []

    try:
        import torch

        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert("RGB")
        else:
            img = image_path_or_pil.convert("RGB")

        w, h = img.size
        tensor = transforms(img).unsqueeze(0)

        with torch.no_grad():
            preds = model(tensor)[0]

        boxes = preds["boxes"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        birds = []
        non_birds = []

        for box, label, score in zip(boxes, labels, scores):
            score_val = float(score)
            label_idx = int(label)
            cat_name = categories[label_idx] if label_idx < len(categories) else str(label_idx)

            if cat_name == "bird":
                if score_val >= bird_thresh:
                    x1, y1, x2, y2 = box
                    bw, bh = float(x2 - x1), float(y2 - y1)
                    if bw >= 15 and bh >= 15:
                        birds.append({
                            "id": len(birds) + 1,
                            "box": [float(x1), float(y1), float(x2), float(y2)],
                            "score": round(score_val, 3),
                            "bw": bw,
                            "bh": bh,
                        })
            else:
                if score_val >= non_bird_thresh:
                    x1, y1, x2, y2 = box
                    non_birds.append({
                        "label": cat_name,
                        "score": round(score_val, 3),
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                    })

        return birds, non_birds
    except Exception as exc:
        logger.error("Error running specimen detector: %s", exc)
        return [], []


def annotate_flock_image(image_path, detections, output_path):
    """Draw bounding boxes and badges on detected chickens and save to output_path."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Try loading a basic font, fallback to default
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        box = det["box"]
        cid = det["id"]
        pred_label = det.get("predicted_disease", "Chicken")
        is_uncertain = det.get("is_uncertain", False)
        det_score = det.get("score", 0.0)

        x1, y1, x2, y2 = [int(v) for v in box]
        # Keep inside image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Color: Green for Healthy, Red for Diseased, Orange for Uncertain
        if is_uncertain:
            color = "#f59e0b"  # Amber
        elif pred_label.lower() == "healthy":
            color = "#10b981"  # Emerald green
        else:
            color = "#ef4444"  # Red

        # Draw bounding box (line thickness 3)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color
            )

        # Label tag
        badge_text = f"#{cid}: {pred_label} (Det: {det_score*100:.0f}%)"
        if is_uncertain:
            badge_text = f"#{cid}: Uncertain (Det: {det_score*100:.0f}%)"

        text_pad = 4
        # Draw background tag
        tag_y1 = max(0, y1 - 22)
        tag_y2 = y1
        tag_x2 = min(w, x1 + len(badge_text) * 8 + text_pad * 2)
        draw.rectangle([x1, tag_y1, tag_x2, tag_y2], fill=color)
        draw.text((x1 + text_pad, tag_y1 + 2), badge_text, fill="#ffffff", font=font)

    img.save(output_path, "JPEG", quality=92)
    return output_path


def analyze_flock_image(image_path, disease_model, class_names, upload_dir=None):
    """Perform complete multi-chicken flock analysis.
    
    1. Detects chickens using FasterRCNN.
    2. Rejects if no chickens detected (or image is unrelated).
    3. Crops each sufficiently visible chicken.
    4. Runs body-image classifier (MobileNetV2) on each crop.
    5. Flags uncertain cases (small crops, low confidence, low detection score).
    6. Generates annotated image with bounding boxes.
    7. Returns comprehensive flock statistics and per-chicken results.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    birds, non_birds = detect_specimens(img, bird_thresh=0.35, non_bird_thresh=0.35)

    # Check if image has no chickens or is dominated by non-chickens
    if len(birds) == 0:
        dominant_non = non_birds[0]["label"] if non_birds else "unidentified object"
        return {
            "success": False,
            "error": "Invalid specimen image — no chicken detected. Please upload a clear chicken photograph.",
            "detected_non_poultry": dominant_non if non_birds else None,
            "total_chickens": 0,
            "chickens": [],
        }

    # Process each detected chicken
    chicken_results = []
    IMG_SIZE = (224, 224)

    healthy_count = 0
    diseased_count = 0
    uncertain_count = 0

    for ch in birds:
        x1, y1, x2, y2 = ch["box"]
        bw, bh = ch["bw"], ch["bh"]

        # Slight 5% padding around bounding box for context
        pad_x = max(4, int(bw * 0.05))
        pad_y = max(4, int(bh * 0.05))
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))

        crop = img.crop((cx1, cy1, cx2, cy2))

        # Check if crop is too small or obscured
        too_small = (bw < 50 or bh < 50)
        low_detection = (ch["score"] < 0.45)

        # Preprocess crop matching exact model input: Bilinear resize (224,224), rescale [0,1]
        crop_resized = crop.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(crop_resized, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # Run inference using body-image classifier
        pred_probs = disease_model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(pred_probs))
        pred_class = class_names[top_idx] if top_idx < len(class_names) else f"Class_{top_idx}"
        class_conf = float(pred_probs[top_idx]) * 100.0

        # Margin & uncertainty
        sorted_probs = np.sort(pred_probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

        is_uncertain = (
            too_small or
            low_detection or
            (class_conf < 65.0) or
            (margin < 0.20)
        )

        uncertainty_reason = ""
        if too_small:
            uncertainty_reason = "Small or distant bird crop"
        elif low_detection:
            uncertainty_reason = "Low detection confidence"
        elif class_conf < 65.0 or margin < 0.20:
            uncertainty_reason = "Ambiguous disease feature distribution"

        if is_uncertain:
            uncertain_count += 1
        elif pred_class.lower() == "healthy":
            healthy_count += 1
        else:
            diseased_count += 1

        prob_breakdown = {
            class_names[i]: round(float(pred_probs[i]) * 100.0, 1)
            for i in range(min(len(class_names), len(pred_probs)))
        }

        chicken_results.append({
            "id": ch["id"],
            "box": [round(v, 1) for v in ch["box"]],
            "crop_dimensions": f"{int(bw)}x{int(bh)}",
            "detection_confidence": round(ch["score"] * 100.0, 1),
            "predicted_disease": pred_class,
            "classification_confidence": round(class_conf, 1),
            "is_uncertain": is_uncertain,
            "uncertainty_reason": uncertainty_reason,
            "probabilities": prob_breakdown,
        })

    # Save annotated image
    base_name = os.path.basename(image_path)
    annotated_filename = f"annotated_{base_name}"
    if upload_dir:
        annotated_path = os.path.join(upload_dir, annotated_filename)
    else:
        annotated_path = os.path.join(os.path.dirname(image_path), annotated_filename)

    annotate_flock_image(image_path, chicken_results, annotated_path)

    return {
        "success": True,
        "total_chickens": len(birds),
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "uncertain_count": uncertain_count,
        "chickens": chicken_results,
        "annotated_filename": annotated_filename,
        "annotated_path": annotated_path,
        "veterinary_disclaimer": (
            "Veterinary Screening Disclaimer: Multi-Chicken flock screening is an automated "
            "computer vision tool. It does not replace physical examination, microbiological "
            "culture, or PCR confirmation by a certified avian veterinarian. Predictions for "
            "distant, obscured, or motion-blurred birds may require close-up individual inspection."
        ),
    }
