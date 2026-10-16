"""PoultryScan AI — Input Image Validation & Out-of-Distribution Rejection Module.

This module validates uploaded images before they reach the disease prediction model.
It addresses the closed-world softmax vulnerability where non-poultry images
(flowers, cars, people, animals, everyday objects, blank screens) are erroneously
assigned high-confidence disease diagnoses.

Validation Architecture:
1. Structural / Usability Heuristics:
   - Verifies image integrity and readable dimensions.
   - Rejects blank, solid-color, or uniform backgrounds via per-channel spatial standard deviation.
   - Rejects unreadable or severely degraded files.

2. Semantic Non-Poultry Rejection (Pre-trained ImageNet Classifier):
   - Uses pre-trained MobileNetV2 (ImageNet) to identify common out-of-distribution entities:
     vehicles, humans/apparel, flowers/ornamental plants, mammalian pets/predators,
     and consumer household electronics.
   - If an image strongly matches a non-poultry category, it is rejected.
   - Conversely, avian/poultry classes ('hen', 'cock', etc.) provide strong positive evidence.

3. In-Distribution Domain Feature Similarity:
   - Computes cosine similarity between the candidate image's penultimate feature embedding
     (1280-D MobileNetV2 GlobalAveragePooling2D) and precomputed centroids of authentic
     training poultry specimens (both head/face lesions and pathognomonic fecal droppings).
   - If the similarity to all authentic poultry centroids falls below the calibrated threshold,
     the image is rejected as out-of-distribution.

4. Examination Mode Context:
   - In 'head_face' mode, verifies anatomical poultry features.
   - In 'full_body' mode, allows both full avian body posture and fresh fecal deposits.

Fail-Safety Policy:
   - If both the ImageNet semantic classifier AND the domain similarity check are unavailable
     (e.g., weights not cached, network error, TF incompatibility), the image is REJECTED with
     an explicit "validator unavailable" message rather than being silently passed through.
   - The domain similarity check is FAIL-CLOSED: if centroids or feat_model cannot be loaded,
     similarity is treated as 0.0 (rejection), not 1.0 (pass-through).

Known Limitations (do not overclaim accuracy):
   - The ImageNet semantic filter covers common out-of-distribution objects but cannot
     enumerate all possible non-poultry images (e.g., microscope slides, abstract art,
     unusual lighting conditions may not be rejected).
   - The cosine similarity threshold (0.50) was calibrated on the available training set;
     it has NOT been independently validated on a large held-out non-poultry corpus.
   - Neither check guarantees 100%% rejection of adversarial or borderline images.
   - The system must be tested on your specific real-world images before clinical reliance.
"""

import json
import logging
import os
import numpy as np
from PIL import Image

logger = logging.getLogger("PoultryScanAI.Validator")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CENTROIDS_PATH = os.path.join(BASE_DIR, "model", "poultry_feature_centroids.json")

# Calibrated cosine similarity threshold (empirically validated: test poultry min=0.523, non-poultry <0.50)
SIMILARITY_THRESHOLD = 0.50

# Spatial texture threshold for blank/uniform image detection
MIN_SPATIAL_STD_DEV = 8.0

# ImageNet non-poultry label keywords
NON_POULTRY_KEYWORDS = {
    # Vehicles & Transportation
    "car", "sports_car", "racer", "convertible", "minivan", "cab", "limousine",
    "jeep", "pickup", "truck", "tow_truck", "trailer_truck", "bus", "school_bus",
    "bicycle", "mountain_bike", "motorcycle", "motor_scooter", "airliner", "airplane",
    "warplane", "wing", "aircraft_carrier", "ship", "boat", "speed_boat", "lifeboat",
    "submarine", "train", "locomotive", "streetcar", "trolleybus", "tank", "tractor",
    "fire_engine", "garbage_truck", "ambulance",
    
    # Humans, Faces, Apparel & Accessories
    "suit", "groom", "trench_coat", "lab_coat", "cowboy_hat", "sombrero", "mortarboard",
    "wig", "jean", "jersey", "kimono", "swimsuit", "bikini", "brassiere", "military_uniform",
    "scuba_diver", "sweatshirt", "cardigan", "bowtie", "necktie", "sunglasses", "sunglass",
    "diaper", "apron", "cloak", "stole", "fur_coat", "bonnet", "crash_helmet",
    
    # Flora, Flowers & Garden Plants
    "daisy", "sunflower", "rose", "tulip", "yellow_lady's_slipper", "pot", "flowerpot",
    "vase", "cardoon", "cliff_rose", "broccoli", "cauliflower", "cabbage", "zucchini",
    "acorn_squash", "butternut_squash", "cucumber", "artichoke", "bell_pepper",
    "orange", "lemon", "banana", "apple", "strawberry", "pineapple",
    
    # Domestic Mammals & Carnivores
    "dog", "retriever", "terrier", "spaniel", "shepherd", "hound", "bulldog", "poodle",
    "cat", "tabby", "tiger_cat", "persian_cat", "siamese_cat", "lion", "tiger", "cheetah",
    "leopard", "jaguar", "bear", "brown_bear", "polar_bear", "black_bear", "panda",
    "wolf", "coyote", "fox", "horse", "zebra", "elephant", "camel", "hippopotamus",
    "rhinoceros", "gorilla", "chimpanzee", "monkey", "baboon", "pig", "wild_boar",
    "cow", "bull", "ox", "bison", "sheep", "ram", "goat", "deer", "gazelle", "rabbit", "hare",
    
    # Inanimate Consumer Objects, Furniture & Electronics
    "cup", "coffee_mug", "espresso", "teacup", "teapot", "coffeepot", "water_bottle", "wine_bottle",
    "beer_bottle", "plate", "soup_bowl", "laptop", "notebook", "monitor", "screen", "television",
    "desktop_computer", "computer_keyboard", "mouse", "cellular_telephone", "cellphone", "dial_telephone",
    "table", "dining_table", "desk", "chair", "armchair", "folding_chair", "couch", "sofa",
    "bed", "wardrobe", "bookcase", "refrigerator", "microwave", "toaster", "vacuum", "iron",
    "clock", "digital_clock", "digital_watch", "wall_clock", "backpack", "wallet", "purse", "umbrella",
}

# Explicit positive poultry / avian categories in ImageNet
POULTRY_AVIAN_KEYWORDS = {
    "cock", "hen", "chick", "prairie_chicken", "ruffed_grouse", "black_grouse",
    "quail", "partridge", "peacock", "pheasant", "water_ouzel", "ptarmigan",
    "drake", "goose", "duck", "gander"
}

_imagenet_model = None
_cached_centroids = None
_feat_model = None


def get_imagenet_model():
    """Lazy load pre-trained ImageNet MobileNetV2 model once."""
    global _imagenet_model
    if _imagenet_model is None:
        try:
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
            _imagenet_model = MobileNetV2(weights="imagenet")
            logger.info("Pre-trained ImageNet MobileNetV2 validator loaded successfully.")
        except Exception as exc:
            logger.error(
                "VALIDATOR WARNING: Could not load ImageNet MobileNetV2 semantic filter: %s. "
                "Non-poultry images that pass Stage 1 heuristics will be blocked by "
                "domain similarity (Stage 3) or rejected if that also fails.",
                exc
            )
            _imagenet_model = None
    return _imagenet_model


def get_feat_model(disease_model):
    """Retrieve or construct cached penultimate feature extractor from the disease model.

    Tries three strategies in order to handle Keras API differences across versions:
    1. keras.Model with feat_layer.output  (standard Keras 3 / TF 2.x path)
    2. Intermediate layer output via model.output at layer index (fallback)
    3. tensorflow.keras.Model fallback
    """
    global _feat_model
    if _feat_model is not None:
        return _feat_model
    if disease_model is None:
        return None

    # Strategy 1: Standard path — works when get_layer().output is accessible
    try:
        import keras
        feat_layer = disease_model.get_layer("global_avg_pool")
        _feat_model = keras.models.Model(
            inputs=disease_model.inputs,
            outputs=feat_layer.output
        )
        # Verify with a test pass
        import numpy as np
        _feat_model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
        logger.info("Feature extractor constructed (Strategy 1 — keras.Model with layer.output).")
        return _feat_model
    except Exception as exc1:
        logger.debug("feat_model Strategy 1 failed: %s", exc1)
        _feat_model = None

    # Strategy 2: Use layer index in model.layers to find global_avg_pool output
    try:
        import keras
        import tensorflow as tf
        layer_names = [l.name for l in disease_model.layers]
        gap_idx = layer_names.index("global_avg_pool")
        # Build a functional model up to and including that layer
        inp = disease_model.input
        x = inp
        for layer in disease_model.layers[1:gap_idx + 1]:
            x = layer(x)
        _feat_model = keras.models.Model(inputs=inp, outputs=x)
        import numpy as np
        _feat_model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
        logger.info("Feature extractor constructed (Strategy 2 — layer-by-layer functional model).")
        return _feat_model
    except Exception as exc2:
        logger.debug("feat_model Strategy 2 failed: %s", exc2)
        _feat_model = None

    # Strategy 3: tensorflow.keras fallback
    try:
        from tensorflow.keras.models import Model as TFModel
        feat_layer = disease_model.get_layer("global_avg_pool")
        _feat_model = TFModel(inputs=disease_model.inputs, outputs=feat_layer.output)
        import numpy as np
        _feat_model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
        logger.info("Feature extractor constructed (Strategy 3 — tensorflow.keras.Model fallback).")
        return _feat_model
    except Exception as exc3:
        logger.error(
            "VALIDATOR WARNING: All feat_model construction strategies failed. "
            "Domain similarity check (Stage 3) will be unavailable. "
            "Errors: [S1] %s [S2] (see debug log) [S3] %s",
            exc1, exc3
        )
        _feat_model = None
        return None


def validator_status():
    """Return a dict describing which validation subsystems are available.
    
    Call this at application startup for diagnostic logging.
    """
    centroids = load_centroids()
    detector_ok = False
    try:
        from chicken_detector import get_detector
        m, _, _ = get_detector()
        detector_ok = (m is not None)
    except Exception:
        detector_ok = False

    return {
        "imagenet_model_loaded": _imagenet_model is not None,
        "centroids_loaded": centroids is not None,
        "num_centroid_classes": len(centroids) if centroids else 0,
        "centroid_path_exists": os.path.exists(CENTROIDS_PATH),
        "semantic_filter_active": _imagenet_model is not None,
        "domain_similarity_active": (centroids is not None),
        "specimen_detector_active": detector_ok,
    }


def load_centroids():
    """Load precomputed 1280-D reference poultry centroids."""
    global _cached_centroids
    if _cached_centroids is None and os.path.exists(CENTROIDS_PATH):
        try:
            with open(CENTROIDS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _cached_centroids = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
                logger.info("Loaded poultry feature centroids for %d classes.", len(_cached_centroids))
        except Exception as exc:
            logger.warning("Could not load poultry feature centroids: %s", exc)
            _cached_centroids = None
    return _cached_centroids


def check_heuristics(img_pil):
    """Check basic structural and spatial texture properties.
    
    Returns (passed: bool, reason: str or None)
    """
    w, h = img_pil.size
    if w < 40 or h < 40:
        return False, "Image dimensions are too small (minimum 40x40 pixels required)."

    arr = np.asarray(img_pil.convert("RGB"), dtype=np.float32)
    # Check spatial standard deviation within each channel
    spatial_stds = [float(np.std(arr[:, :, c])) for c in range(3)]
    avg_spatial_std = float(np.mean(spatial_stds))

    if avg_spatial_std < MIN_SPATIAL_STD_DEV:
        return False, "Image has insufficient visual detail (blank or uniform solid color)."

    return True, None


def check_imagenet_semantics(img_pil):
    """Classify image against ImageNet to detect non-poultry entities.
    
    Returns (is_rejected: bool, detected_label: str or None, top_prob: float, is_positive_poultry: bool)
    """
    model = get_imagenet_model()
    if model is None:
        return False, None, 0.0, False

    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
        img_resized = img_pil.convert("RGB").resize((224, 224))
        x = np.expand_dims(np.asarray(img_resized, dtype=np.float32), axis=0)
        x = preprocess_input(x)
        preds = model.predict(x, verbose=0)
        top5 = decode_predictions(preds, top=5)[0]

        top_label = top5[0][1].lower()
        top_prob = float(top5[0][2])

        # Check positive poultry indicator
        is_positive_poultry = any(k in top_label for k in POULTRY_AVIAN_KEYWORDS)
        if is_positive_poultry:
            return False, top_label, top_prob, True

        # Check top prediction against non-poultry list
        for _, label, prob in top5[:3]:
            lbl = label.lower().replace("-", "_")
            tokens = set(lbl.split("_"))
            is_non_poultry = bool(tokens.intersection(NON_POULTRY_KEYWORDS) or lbl in NON_POULTRY_KEYWORDS)
            if is_non_poultry and float(prob) >= 0.22:
                return True, label, float(prob), False

        return False, top_label, top_prob, False

    except Exception as exc:
        logger.warning("ImageNet semantic check failed: %s", exc)
        return False, None, 0.0, False


def check_domain_similarity(img_pil, disease_model):
    """Compute cosine similarity of penultimate features against authentic poultry centroids.

    FAIL-CLOSED policy: if centroids or feat_model cannot be loaded, returns (0.0, None)
    instead of (1.0, None). This means an unavailable Stage-3 check REJECTS rather than
    silently passes the image.
    
    Returns (max_similarity: float, closest_class: str or None, check_available: bool)
    """
    centroids = load_centroids()
    if centroids is None:
        logger.warning(
            "Domain similarity unavailable: centroids file not found at %s. "
            "Returning 0.0 (fail-closed).",
            CENTROIDS_PATH
        )
        return 0.0, None, False  # fail-closed: treat as rejected

    if disease_model is None:
        logger.warning("Domain similarity unavailable: disease_model is None. Returning 0.0 (fail-closed).")
        return 0.0, None, False

    feat_model = get_feat_model(disease_model)
    if feat_model is None:
        logger.warning(
            "Domain similarity unavailable: feat_model could not be constructed. "
            "Returning 0.0 (fail-closed)."
        )
        return 0.0, None, False

    try:
        img_resized = img_pil.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        x = np.expand_dims(arr, axis=0)

        emb = feat_model.predict(x, verbose=0)[0]
        norm = np.linalg.norm(emb)
        if norm < 1e-12:
            return 0.0, None, True

        emb_norm = emb / norm
        # Normalise each centroid vector before dot product for correct cosine similarity
        sims = {}
        for cls_name, c_vec in centroids.items():
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 1e-12:
                sims[cls_name] = float(np.dot(emb_norm, c_vec / c_norm))
            else:
                sims[cls_name] = 0.0
        closest_class = max(sims, key=sims.get)
        max_similarity = sims[closest_class]
        return max_similarity, closest_class, True

    except Exception as exc:
        logger.warning("Domain similarity computation failed: %s. Returning 0.0 (fail-closed).", exc)
        return 0.0, None, False


def validate_poultry_image(image_path, image_type="full_body", disease_model=None):
    """Run comprehensive multi-layer validation on a candidate image.
    
    Args:
        image_path: Path to the image file on disk.
        image_type: Examination mode ('head_face' or 'full_body').
        disease_model: Loaded Keras model instance (used for domain similarity).

    Returns:
        is_valid (bool): True if the image is a suitable poultry image.
        message (str): User-facing message (empty if valid, clear explanation if invalid).
        audit_details (dict): Quantitative metrics and signals for logging/testing.

    Fail-safety:
        If both the ImageNet semantic check AND domain similarity are unavailable
        (i.e., neither subsystem could load its model), the image is REJECTED with
        a clear explanation. This prevents non-poultry images from slipping through
        due to infrastructure failures.
    """
    if not os.path.exists(image_path):
        return False, "Invalid input: Image file not found.", {"error": "file_not_found"}

    try:
        img_pil = Image.open(image_path)
    except Exception as exc:
        return False, "Invalid input: Could not read image file.", {"error": str(exc)}

    # Stage 1: Structural & Texture Heuristics
    h_passed, h_reason = check_heuristics(img_pil)
    if not h_passed:
        logger.info("Heuristic rejection for %s: %s", image_path, h_reason)
        return False, "Invalid specimen image — no chicken detected. Please upload a clear chicken photograph.", {
            "stage": "heuristics",
            "reason": h_reason
        }

    # Stage 1B: Specimen Object Detector (FasterRCNN chicken presence & non-poultry detection)
    PROMINENT_NON_POULTRY_LABELS = {
        "person", "car", "truck", "bus", "train", "motorcycle", "bicycle", "airplane",
        "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "boat", "sports ball", "tennis racket", "baseball glove", "skateboard", "surfboard",
    }
    try:
        from chicken_detector import detect_specimens
        birds, non_birds = detect_specimens(img_pil, bird_thresh=0.30, non_bird_thresh=0.45)
    except Exception as det_exc:
        logger.warning("Chicken detector check skipped: %s", det_exc)
        birds, non_birds = [], []

    has_detected_chicken = (len(birds) > 0)

    # In head_face mode: reject if human, vehicle, or sports entity is detected (e.g. cricket field person)
    # but do not reject close-up chicken comb/wattles due to COCO animal confusion
    if image_type == "head_face":
        prominent_non_birds = [
            nb for nb in non_birds
            if nb["label"] in {"person", "car", "truck", "bus", "motorcycle", "bicycle", "sports ball", "tennis racket"}
            and nb["score"] >= 0.50
        ]
    else:
        prominent_non_birds = [
            nb for nb in non_birds
            if nb["label"] in PROMINENT_NON_POULTRY_LABELS and nb["score"] >= 0.50
        ]

    # If prominent human, vehicle, predator, or sports equipment is detected without any chicken:
    if len(prominent_non_birds) > 0 and not has_detected_chicken:
        dominant_label = prominent_non_birds[0]["label"].replace("_", " ").title()
        dominant_score = prominent_non_birds[0]["score"]
        logger.info(
            "Detector rejection for %s: detected %s (%.1f%%), no chickens detected",
            image_path, dominant_label, dominant_score * 100
        )
        return False, "Invalid specimen image — no chicken detected. Please upload a clear chicken photograph.", {
            "stage": "specimen_detector",
            "detected_non_poultry": dominant_label,
            "confidence": round(dominant_score * 100, 2),
            "birds_detected": 0
        }

    # Stage 2: Semantic Non-Poultry Rejection (ImageNet MobileNetV2)
    imagenet_available = (get_imagenet_model() is not None)
    is_rejected, detected_lbl, sem_prob, is_poultry_avian = check_imagenet_semantics(img_pil)
    if is_rejected:
        readable_label = detected_lbl.replace("_", " ").title()
        msg = "Invalid specimen image — no chicken detected. Please upload a clear chicken photograph."
        logger.info("Semantic rejection for %s: %s (%.1f%%)", image_path, detected_lbl, sem_prob * 100)
        return False, msg, {
            "stage": "semantic_filter",
            "detected_label": detected_lbl,
            "confidence": round(sem_prob * 100, 2),
            "is_poultry_avian": is_poultry_avian
        }

    # Stage 3: Domain Feature Consistency (Cosine similarity against poultry centroids)
    max_sim, closest_class, sim_available = check_domain_similarity(img_pil, disease_model)
    logger.info(
        "Domain similarity for %s: %.3f (closest: %s, check_available: %s)",
        image_path, max_sim, closest_class, sim_available
    )

    # Fail-safety guard
    if not imagenet_available and not sim_available and not has_detected_chicken:
        msg = (
            "Invalid input: The image validator subsystems could not be loaded "
            "(ImageNet semantic filter and domain similarity both unavailable). "
            "Please ensure the application has internet access for first-time setup, "
            "then restart the server. If the problem persists, contact the administrator."
        )
        logger.error(
            "VALIDATOR FAIL-SAFE TRIGGERED for %s: Stage 2, Stage 3 and detector unavailable.",
            image_path
        )
        return False, msg, {
            "stage": "fail_safe",
            "imagenet_available": imagenet_available,
            "sim_available": sim_available,
        }

    # Reject if features deviate from poultry manifold (unless chicken was confirmed by detector/ImageNet)
    if (not is_poultry_avian) and (not has_detected_chicken) and sim_available and (max_sim < SIMILARITY_THRESHOLD):
        msg = "Invalid specimen image — no chicken detected. Please upload a clear chicken photograph."
        logger.info(
            "Domain similarity rejection for %s: sim=%.3f < threshold=%.2f (closest: %s)",
            image_path, max_sim, SIMILARITY_THRESHOLD, closest_class
        )
        return False, msg, {
            "stage": "domain_similarity",
            "max_similarity": round(max_sim, 3),
            "threshold": SIMILARITY_THRESHOLD,
            "closest_class": closest_class,
            "is_poultry_avian": is_poultry_avian
        }

    return True, "", {
        "stage": "passed",
        "max_similarity": round(max_sim, 3) if sim_available else None,
        "closest_class": closest_class,
        "is_poultry_avian": is_poultry_avian,
        "detected_semantic": detected_lbl,
        "imagenet_available": imagenet_available,
        "sim_available": sim_available,
    }
