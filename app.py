"""PoultryScan AI — Complete Poultry Disease Detection Flask Application.

Architecture: MobileNetV2 Transfer Learning (.keras model)
Supported Five Classes:
  1. Coccidiosis
  2. Fowl Pox
  3. Healthy
  4. Newcastle Disease
  5. Salmonella

Database: SQLite (predictions.db)
Examination Modes:
  - Chicken Head / Face Image Analysis
  - Full Chicken Body / Droppings Image Analysis
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime

import numpy as np
from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from diseases import (
    ACTIVE_MODEL_CLASSES,
    CLASS_NAMES,
    DISEASE_INFO,
    REFERENCE_CLASSES,
)
from image_validator import (
    get_imagenet_model,
    load_centroids,
    validate_poultry_image,
    validator_status,
)
from chicken_detector import analyze_flock_image, get_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PoultryScanAI")

# Directory paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# On Hugging Face Spaces, /data is a persistent volume that survives restarts.
# Use it for uploads and the SQLite database so scan history is not lost.
# On local dev (no /data), fall back to project-local directories.
_HF_DATA = "/data"
_use_hf_data = os.path.isdir(_HF_DATA) and os.access(_HF_DATA, os.W_OK)

if _use_hf_data:
    UPLOAD_DIR = os.path.join(_HF_DATA, "uploads")
    DATABASE_DIR = os.path.join(_HF_DATA, "database")
    logger.info("Using HF Spaces persistent /data volume for uploads and database.")
else:
    UPLOAD_DIR = os.path.abspath(os.path.join(BASE_DIR, "uploads"))
    DATABASE_DIR = os.path.abspath(os.path.join(BASE_DIR, "database"))
    logger.info("Using local directories for uploads and database.")

DB_PATH = os.path.join(DATABASE_DIR, "predictions.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "poultry_disease_model.keras")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "model", "class_indices.json")
EVAL_SUMMARY_PATH = os.path.join(BASE_DIR, "evaluation_results", "evaluation_summary.json")
GRADCAM_DIR = os.path.abspath(os.path.join(BASE_DIR, "research_results", "grad_cam_examples"))
GRADCAM_MANIFEST_PATH = os.path.join(GRADCAM_DIR, "gradcam_manifest.json")

# Supported file formats and upload limits
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB maximum upload

# Model configuration
IMG_SIZE = (224, 224)              # MobileNetV2 input spatial dimensions
CONFIDENCE_THRESHOLD = 65.0        # Percent below which a screening is flagged as uncertain
UNCERTAIN_ADVISORY = (
    "Diagnostic result shows low statistical confidence or ambiguous classification. "
    "This may indicate an atypical specimen or early-stage pathology. "
    "Please consult a certified poultry veterinarian for clinical laboratory confirmation."
)

# Canonical label aliases — normalize any spelling variants before DB writes or display
LABEL_ALIASES = {
    "New Castle Disease": "Newcastle Disease",
    "new castle disease": "Newcastle Disease",
    "newcastle disease": "Newcastle Disease",
    "salmonellosis": "Salmonella",
    "Salmonellosis": "Salmonella",
    "healthy chicken": "Healthy",
    "Healthy Chicken": "Healthy",
}

# Examination modes
VALID_IMAGE_TYPES = {
    "head_face": "Chicken Head / Face Image",
    "full_body": "Full Chicken Body Image",
}

# Ensure essential storage directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)

# Initialize Flask
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "poultry-scan-ai-engineering-secret-key-2026")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# ==============================================================================
# Dynamic Class Mapping & Model Loading
# ==============================================================================
def load_class_names():
    """Load exact class mapping from class_indices.json, falling back to diseases.py."""
    if os.path.exists(CLASS_INDICES_PATH):
        try:
            with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    logger.info("Loaded %d class names from class_indices.json: %s", len(data), data)
                    return data
                elif isinstance(data, dict):
                    loaded = [data[str(i)] for i in range(len(data))]
                    logger.info("Loaded %d class names from dict class_indices.json: %s", len(loaded), loaded)
                    return loaded
        except Exception as exc:
            logger.warning("Could not parse class_indices.json: %s", exc)
    return CLASS_NAMES


CURRENT_CLASSES = load_class_names()
app.config["CLASS_NAMES"] = CURRENT_CLASSES


def load_trained_model():
    """Load the trained .keras model safely and cache in app config.

    Validates model existence, input shape, and output neurons against CURRENT_CLASSES.
    Avoids mock/fabricated predictions.
    """
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at expected path: %s", MODEL_PATH)
        return None

    try:
        try:
            import keras
            model = keras.models.load_model(MODEL_PATH, compile=False)
        except Exception:
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_PATH, compile=False)

        output_dim = model.output_shape[-1]
        expected_dim = len(CURRENT_CLASSES)
        if output_dim != expected_dim:
            logger.warning(
                "Model output dimension (%d) does not match CURRENT_CLASSES (%d). Using available neurons.",
                output_dim, expected_dim
            )

        # Verify inference works end-to-end with a blank test tensor
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        test_out = model.predict(dummy, verbose=0)
        if test_out is None or len(test_out[0]) != output_dim:
            raise RuntimeError("Inference test returned unexpected output shape.")

        logger.info(
            "Trained model loaded and inference-tested from %s (Output neurons: %d)",
            MODEL_PATH, output_dim
        )
        return model

    except Exception as exc:
        logger.exception("Failed to load model from %s: %s", MODEL_PATH, exc)
        return None


# Cache model at application startup
app.config["MODEL"] = load_trained_model()

# Preload poultry validation resources at startup for fast inference
load_centroids()
get_imagenet_model()
_vstatus = validator_status()
logger.info(
    "[VALIDATOR STATUS] ImageNet semantic filter: %s | Domain similarity: %s (%d centroid classes) | "
    "Centroid file present: %s",
    'ACTIVE' if _vstatus['semantic_filter_active'] else 'UNAVAILABLE (non-poultry images will be blocked by domain similarity or fail-safe)',
    'ACTIVE' if _vstatus['domain_similarity_active'] else 'UNAVAILABLE',
    _vstatus['num_centroid_classes'],
    _vstatus['centroid_path_exists'],
)


# ==============================================================================
# Image Validation & Preprocessing
# ==============================================================================
def allowed_file(filename):
    """Check if the filename has an allowed extension (JPG, JPEG, PNG, WEBP)."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_file(file_storage):
    """Validate upload presence, filename extension, and decode header with Pillow."""
    if file_storage is None or file_storage.filename == "":
        return False, "Please select an image file to upload."

    filename = file_storage.filename
    if not allowed_file(filename):
        return False, "Invalid file format. Only JPG, JPEG, PNG, and WEBP images are accepted."

    try:
        image = Image.open(file_storage.stream)
        image.verify()  # Verifies file integrity and format header
        file_storage.stream.seek(0)
    except (UnidentifiedImageError, OSError, Exception):
        return False, "The uploaded file is corrupt or not a readable image."

    return True, None


def save_secure_upload(file_storage):
    """Save uploaded image into uploads/ folder with sanitized, unique filename."""
    clean_name = secure_filename(file_storage.filename) or "sample.jpg"
    unique_prefix = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    saved_filename = f"{unique_prefix}_{clean_name}"

    dest_path = os.path.abspath(os.path.join(UPLOAD_DIR, saved_filename))
    if not dest_path.startswith(UPLOAD_DIR):
        raise ValueError("Invalid target path detected. Upload aborted.")

    file_storage.save(dest_path)
    return saved_filename, dest_path


def preprocess_image(image_path):
    """Apply exact preprocessing matching model training:
    1. Convert to RGB.
    2. Bilinear resize to (224, 224).
    3. Rescale pixel values to [0.0, 1.0].
    4. Expand batch dimension to (1, 224, 224, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def evaluate_uncertainty(probabilities, image_type="full_body"):
    """Evaluate statistical uncertainty based on confidence, margin, and entropy."""
    sorted_probs = np.sort(probabilities)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top1 - top2
    num_classes = len(probabilities)
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)) / np.log(num_classes))

    is_uncertain = (
        (top1 < (CONFIDENCE_THRESHOLD / 100.0)) or
        (margin < 0.20) or
        (entropy > 0.65)
    )
    return is_uncertain, round(entropy, 3), round(margin * 100.0, 2)


def normalize_label(label):
    """Normalize a disease label to its canonical form using LABEL_ALIASES."""
    return LABEL_ALIASES.get(label, label)


def predict_image(image_path, image_type="full_body"):
    """Run model inference on the preprocessed image with uncertainty checks."""
    model = app.config.get("MODEL")
    if model is None:
        raise RuntimeError("Trained model is not loaded. Please ensure model/poultry_disease_model.keras is present.")

    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)
    input_tensor = preprocess_image(image_path)
    raw_predictions = model.predict(input_tensor, verbose=0)
    probabilities = raw_predictions[0]

    top_index = int(np.argmax(probabilities))
    raw_class = classes[top_index] if top_index < len(classes) else f"Class_{top_index}"
    # Normalize to canonical label
    predicted_class = normalize_label(raw_class)
    confidence_pct = round(float(probabilities[top_index]) * 100.0, 2)

    prob_dict = {
        normalize_label(classes[i]): round(float(probabilities[i]) * 100.0, 2)
        for i in range(min(len(classes), len(probabilities)))
    }

    is_uncertain, entropy_val, margin_val = evaluate_uncertainty(probabilities, image_type=image_type)

    logger.info(
        "Inference: %s (%.2f%%) | Mode: %s | Margin: %.2f%% | Entropy: %.3f | Uncertain: %s",
        predicted_class, confidence_pct, image_type, margin_val, entropy_val, is_uncertain
    )

    return predicted_class, confidence_pct, prob_dict, is_uncertain


# ==============================================================================
# Database Persistence (SQLite)
# ==============================================================================
def get_db():
    """Retrieve or initialize SQLite database connection for current request."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    """Close database connection at end of request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Initialize database schema with migration safeguards."""
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_uncertain INTEGER NOT NULL DEFAULT 0,
                image_type TEXT DEFAULT 'full_body',
                details TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )"""
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS flock_chickens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                chicken_num INTEGER NOT NULL,
                box_coords TEXT NOT NULL,
                crop_size TEXT NOT NULL,
                detection_conf REAL NOT NULL,
                disease TEXT NOT NULL,
                classification_conf REAL NOT NULL,
                is_uncertain INTEGER NOT NULL DEFAULT 0,
                uncertainty_reason TEXT DEFAULT '',
                probabilities TEXT DEFAULT '',
                FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
            )"""
        )
        for col, col_def in [
            ("is_uncertain", "INTEGER NOT NULL DEFAULT 0"),
            ("image_type", "TEXT DEFAULT 'full_body'"),
            ("details", "TEXT DEFAULT ''"),
            ("scan_mode", "TEXT DEFAULT 'single'"),
            ("annotated_filename", "TEXT DEFAULT ''"),
            ("chicken_count", "INTEGER DEFAULT 1"),
        ]:
            try:
                db.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_def}")
            except sqlite3.OperationalError:
                pass


init_db()


# ==============================================================================
# Routes & Endpoints
# ==============================================================================

@app.route("/")
def index():
    """Home page with hero section, KPI metrics, and disease previews."""
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    healthy = db.execute("SELECT COUNT(*) c FROM predictions WHERE disease = 'Healthy'").fetchone()["c"]
    diseased = total - healthy
    model_loaded = (app.config.get("MODEL") is not None)
    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)

    # Load verified evaluation summary — never fabricate metrics
    eval_data = None
    if os.path.exists(EVAL_SUMMARY_PATH):
        try:
            with open(EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            eval_data = None

    return render_template(
        "index.html",
        total=total,
        healthy=healthy,
        diseased=diseased,
        model_loaded=model_loaded,
        class_names=classes,
        active_classes=classes,
        reference_classes=REFERENCE_CLASSES,
        disease_info=DISEASE_INFO,
        eval_data=eval_data,
    )


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard with live charts and summary statistics."""
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    avg = db.execute("SELECT AVG(confidence) a FROM predictions").fetchone()["a"] or 0
    stats = db.execute("SELECT disease, COUNT(*) c FROM predictions GROUP BY disease").fetchall()

    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)
    counts = {c: 0 for c in classes}
    for r in stats:
        canonical = normalize_label(r["disease"])
        if canonical in counts:
            counts[canonical] += r["c"]

    history_records = db.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 10"
    ).fetchall()

    healthy = counts.get("Healthy", 0)
    diseased = total - healthy
    model_loaded = (app.config.get("MODEL") is not None)

    flock_count = db.execute("SELECT COUNT(*) c FROM predictions WHERE scan_mode = 'flock'").fetchone()["c"]
    single_count = total - flock_count

    # Load verified evaluation summary — never fabricate metrics
    eval_data = None
    if os.path.exists(EVAL_SUMMARY_PATH):
        try:
            with open(EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            eval_data = None

    return render_template(
        "dashboard.html",
        total=total,
        single_count=single_count,
        flock_count=flock_count,
        avg=round(avg, 1),
        counts=counts,
        history=history_records,
        healthy=healthy,
        diseased=diseased,
        model_loaded=model_loaded,
        class_names=classes,
        eval_data=eval_data,
    )


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Disease detection page with drag-and-drop upload and file picker."""
    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)
    model_loaded = (app.config.get("MODEL") is not None)

    if request.method == "GET":
        selected_type = request.args.get("type", "full_body")
        if selected_type not in VALID_IMAGE_TYPES:
            selected_type = "full_body"
        return render_template(
            "upload.html",
            model_loaded=model_loaded,
            selected_type=selected_type,
            valid_image_types=VALID_IMAGE_TYPES,
            active_classes=classes,
            reference_classes=REFERENCE_CLASSES,
        )

    # 1. Validate file presence
    uploaded_file = request.files.get("image")
    is_valid, error_msg = validate_image_file(uploaded_file)
    if not is_valid:
        flash(error_msg, "error")
        return redirect(url_for("upload"))

    # 2. Extract examination mode
    image_type = request.form.get("image_type", "full_body").strip().lower()
    if image_type not in VALID_IMAGE_TYPES:
        image_type = "full_body"

    # 3. Model readiness check
    if not model_loaded:
        flash(
            "Prediction service unavailable: Trained model is not loaded. "
            "Please run 'python train_model.py' to generate a verified 5-class model.",
            "error"
        )
        return redirect(url_for("upload"))

    # 4. Save upload securely
    try:
        saved_filename, file_path = save_secure_upload(uploaded_file)
    except Exception as exc:
        logger.error("File save error: %s", exc)
        flash("Could not save the uploaded image securely. Please try again.", "error")
        return redirect(url_for("upload"))

    # 5. Validate suitable poultry image before running disease diagnosis
    model = app.config.get("MODEL")
    is_poultry, validation_error, val_details = validate_poultry_image(
        file_path, image_type=image_type, disease_model=model
    )
    if not is_poultry:
        logger.warning(
            "Rejected non-poultry upload '%s': %s | Details: %s",
            saved_filename, validation_error, val_details
        )
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        flash(validation_error, "error")
        return redirect(url_for("upload"))

    # 6. Run real model inference
    try:
        label, confidence, prob_dict, is_uncertain = predict_image(file_path, image_type=image_type)
    except Exception as exc:
        logger.exception("Inference error on file %s: %s", file_path, exc)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        flash(f"Inference failed while processing the image: {str(exc)}", "error")
        return redirect(url_for("upload"))

    # 6. Save prediction in database
    db = get_db()
    cur = db.execute(
        "INSERT INTO predictions (filename, disease, confidence, is_uncertain, image_type, details, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            saved_filename,
            label,
            confidence,
            int(is_uncertain),
            image_type,
            json.dumps(prob_dict),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    db.commit()

    if is_uncertain:
        flash("Screening completed with low confidence. Please review the veterinary advisory notice.", "warning")
    else:
        flash("Diagnostic screening completed successfully!", "success")

    return redirect(url_for("result", pred_id=cur.lastrowid))


@app.route("/result/<int:pred_id>")
def result(pred_id):
    """Prediction results page showing class, confidence, uploaded image, and disease profile."""
    row = get_db().execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    if row is None:
        flash("The requested diagnostic record was not found.", "error")
        return redirect(url_for("upload"))

    disease_name = row["disease"]
    # Handle aliases
    info = DISEASE_INFO.get(disease_name, DISEASE_INFO.get("Healthy", {}))

    prob_dict = {}
    if row["details"]:
        try:
            prob_dict = json.loads(row["details"])
        except Exception:
            prob_dict = {}

    image_type_key = row["image_type"] if "image_type" in row.keys() else "full_body"
    image_type_label = VALID_IMAGE_TYPES.get(image_type_key, "Full Chicken Body Image")
    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)

    return render_template(
        "result.html",
        pred=row,
        info=info,
        prob_dict=prob_dict,
        class_names=classes,
        image_type_key=image_type_key,
        image_type_label=image_type_label,
        uncertain_advisory=UNCERTAIN_ADVISORY,
    )


@app.route("/multi-chicken", methods=["GET", "POST"])
def multi_chicken():
    """Multi-chicken flock detection, bounding box annotation, and crop health screening."""
    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)
    model = app.config.get("MODEL")
    model_loaded = (model is not None)

    if request.method == "GET":
        return render_template(
            "multi_chicken.html",
            model_loaded=model_loaded,
            active_classes=classes,
            reference_classes=REFERENCE_CLASSES,
        )

    # 1. Validate file presence
    uploaded_file = request.files.get("image")
    is_valid, error_msg = validate_image_file(uploaded_file)
    if not is_valid:
        flash(error_msg, "error")
        return redirect(url_for("multi_chicken"))

    # 2. Model readiness check
    if not model_loaded:
        flash("Prediction service unavailable: Trained model is not loaded.", "error")
        return redirect(url_for("multi_chicken"))

    # 3. Save upload securely
    try:
        saved_filename, file_path = save_secure_upload(uploaded_file)
    except Exception as exc:
        logger.error("File save error: %s", exc)
        flash("Could not save the uploaded image securely. Please try again.", "error")
        return redirect(url_for("multi_chicken"))

    # 4. Run flock analysis
    try:
        flock_res = analyze_flock_image(file_path, model, classes, upload_dir=UPLOAD_DIR)
    except Exception as exc:
        logger.exception("Flock analysis error on %s: %s", file_path, exc)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        flash(f"Flock analysis failed: {str(exc)}", "error")
        return redirect(url_for("multi_chicken"))

    # If no chickens detected, reject upload
    if not flock_res.get("success"):
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        err_msg = flock_res.get(
            "error",
            "Invalid specimen image — no chicken detected. Please upload a clear chicken photograph."
        )
        flash(err_msg, "error")
        return redirect(url_for("multi_chicken"))

    # 5. Save flock scan record in DB
    db = get_db()
    num_chickens = flock_res["total_chickens"]
    all_confs = [ch["classification_confidence"] for ch in flock_res["chickens"]]
    avg_conf = round(float(np.mean(all_confs)), 2) if all_confs else 85.0
    is_flock_uncertain = 1 if flock_res["uncertain_count"] > 0 else 0

    if flock_res["diseased_count"] > 0:
        flock_title = f"Flock Screening: {flock_res['diseased_count']} Suspect, {flock_res['healthy_count']} Healthy"
    else:
        flock_title = f"Flock Screening: {num_chickens} Healthy Birds"

    cur = db.execute(
        "INSERT INTO predictions (filename, disease, confidence, is_uncertain, image_type, scan_mode, "
        "annotated_filename, chicken_count, details, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            saved_filename,
            flock_title,
            avg_conf,
            is_flock_uncertain,
            "full_body",
            "flock",
            flock_res["annotated_filename"],
            num_chickens,
            json.dumps(flock_res),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    pred_id = cur.lastrowid

    # Insert individual chicken detections
    for ch in flock_res["chickens"]:
        db.execute(
            "INSERT INTO flock_chickens (prediction_id, chicken_num, box_coords, crop_size, detection_conf, "
            "disease, classification_conf, is_uncertain, uncertainty_reason, probabilities) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pred_id,
                ch["id"],
                json.dumps(ch["box"]),
                ch["crop_dimensions"],
                ch["detection_confidence"],
                ch["predicted_disease"],
                ch["classification_confidence"],
                1 if ch["is_uncertain"] else 0,
                ch.get("uncertainty_reason", ""),
                json.dumps(ch["probabilities"]),
            ),
        )
    db.commit()

    flash(f"Flock scan completed: {num_chickens} chickens detected.", "success")
    return redirect(url_for("flock_result", pred_id=pred_id))


@app.route("/flock-result/<int:pred_id>")
def flock_result(pred_id):
    """Flock scan results page showing annotated image and individual chicken breakdowns."""
    db = get_db()
    pred = db.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    if pred is None:
        flash("The requested flock screening record was not found.", "error")
        return redirect(url_for("multi_chicken"))

    # Retrieve individual chickens
    ch_rows = db.execute(
        "SELECT * FROM flock_chickens WHERE prediction_id = ? ORDER BY chicken_num ASC",
        (pred_id,)
    ).fetchall()

    chickens = []
    if ch_rows:
        for r in ch_rows:
            probs = json.loads(r["probabilities"]) if r["probabilities"] else {}
            chickens.append({
                "id": r["chicken_num"],
                "box": json.loads(r["box_coords"]) if r["box_coords"] else [],
                "crop_dimensions": r["crop_size"],
                "detection_confidence": r["detection_conf"],
                "predicted_disease": r["disease"],
                "classification_confidence": r["classification_conf"],
                "is_uncertain": bool(r["is_uncertain"]),
                "uncertainty_reason": r["uncertainty_reason"],
                "probabilities": probs,
            })
    elif pred["details"]:
        try:
            det_data = json.loads(pred["details"])
            chickens = det_data.get("chickens", [])
        except Exception:
            chickens = []

    healthy_count = sum(1 for c in chickens if not c["is_uncertain"] and c["predicted_disease"] == "Healthy")
    diseased_count = sum(1 for c in chickens if not c["is_uncertain"] and c["predicted_disease"] != "Healthy")
    uncertain_count = sum(1 for c in chickens if c["is_uncertain"])

    annotated_fn = (
        pred["annotated_filename"]
        if ("annotated_filename" in pred.keys() and pred["annotated_filename"])
        else pred["filename"]
    )

    return render_template(
        "flock_result.html",
        pred=pred,
        annotated_filename=annotated_fn,
        total_chickens=len(chickens),
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        uncertain_count=uncertain_count,
        chickens=chickens,
    )


@app.route("/diseases")
def diseases():
    """Disease information encyclopedia page covering all 5 active diseases + reference conditions."""
    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)
    return render_template(
        "diseases.html",
        diseases=DISEASE_INFO,
        class_names=classes,
        active_classes=classes,
        reference_classes=REFERENCE_CLASSES,
    )


@app.route("/history")
def history():
    """Prediction history page with search and filtering."""
    db = get_db()
    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)
    filter_disease = request.args.get("disease", "").strip()
    filter_type = request.args.get("type", "").strip()
    search_q = request.args.get("q", "").strip()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if filter_disease and (filter_disease in classes or filter_disease in DISEASE_INFO):
        query += " AND disease = ?"
        params.append(filter_disease)

    if filter_type and filter_type in VALID_IMAGE_TYPES:
        query += " AND image_type = ?"
        params.append(filter_type)

    if search_q:
        query += " AND (filename LIKE ? OR disease LIKE ?)"
        params.extend([f"%{search_q}%", f"%{search_q}%"])

    query += " ORDER BY id DESC"
    rows = db.execute(query, params).fetchall()
    total_records = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]

    return render_template(
        "history.html",
        records=rows,
        total_records=total_records,
        filter_disease=filter_disease,
        filter_type=filter_type,
        search_q=search_q,
        class_names=classes,
        valid_image_types=VALID_IMAGE_TYPES,
    )


@app.route("/history/delete/<int:pred_id>", methods=["POST"])
def delete_history(pred_id):
    """Delete a specific record, its annotated crop, and associated flock entries."""
    db = get_db()
    row = db.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    if row:
        target_path = os.path.join(UPLOAD_DIR, row["filename"])
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
        if "annotated_filename" in row.keys() and row["annotated_filename"]:
            ann_path = os.path.join(UPLOAD_DIR, row["annotated_filename"])
            if os.path.exists(ann_path):
                try:
                    os.remove(ann_path)
                except OSError:
                    pass
        db.execute("DELETE FROM flock_chickens WHERE prediction_id = ?", (pred_id,))
        db.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
        db.commit()
        flash(f"Record #{pred_id} deleted successfully.", "success")
    return redirect(url_for("history"))


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Clear all records and associated image files."""
    db = get_db()
    rows = db.execute("SELECT * FROM predictions").fetchall()
    for r in rows:
        target_path = os.path.join(UPLOAD_DIR, r["filename"])
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
        if "annotated_filename" in r.keys() and r["annotated_filename"]:
            ann_path = os.path.join(UPLOAD_DIR, r["annotated_filename"])
            if os.path.exists(ann_path):
                try:
                    os.remove(ann_path)
                except OSError:
                    pass
    db.execute("DELETE FROM flock_chickens")
    db.execute("DELETE FROM predictions")
    db.commit()
    flash("All prediction history cleared.", "success")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    """About project page explaining AI methodology, transfer learning, and limitations."""
    model_loaded = (app.config.get("MODEL") is not None)
    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)

    eval_data = None
    if os.path.exists(EVAL_SUMMARY_PATH):
        try:
            with open(EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            eval_data = None

    gradcam_figures = []
    gradcam_disclaimer = ""
    if os.path.exists(GRADCAM_MANIFEST_PATH):
        try:
            with open(GRADCAM_MANIFEST_PATH, "r", encoding="utf-8") as f:
                gc_data = json.load(f)
                gradcam_disclaimer = gc_data.get("grad_cam_disclaimer", "")
                for fig in gc_data.get("figures", []):
                    fig_copy = dict(fig)
                    fig_copy["basename"] = os.path.basename(fig["output_figure"])
                    gradcam_figures.append(fig_copy)
        except Exception:
            gradcam_figures = []

    return render_template(
        "about.html",
        model_loaded=model_loaded,
        eval_data=eval_data,
        gradcam_figures=gradcam_figures,
        gradcam_disclaimer=gradcam_disclaimer,
        active_classes=classes,
        reference_classes=REFERENCE_CLASSES,
    )


@app.route("/contact", methods=["GET", "POST"])
def contact():
    """Contact & feedback section for farm inquiries and veterinary collaboration."""
    model_loaded = (app.config.get("MODEL") is not None)
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        category = request.form.get("category", "General Feedback").strip()
        message = request.form.get("message", "").strip()
        flash(f"Thank you, {name or 'Partner'}! Your {category} message has been recorded.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html", model_loaded=model_loaded)


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.route("/health")
def health():
    """Health-check route to verify backend readiness and model status.

    Performs a real inference test (blank tensor) so the status reflects
    actual model usability, not just file presence.
    """
    model = app.config.get("MODEL")
    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)
    model_file_exists = os.path.exists(MODEL_PATH)
    inference_ok = False
    inference_error = None

    if model is not None:
        try:
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            out = model.predict(dummy, verbose=0)
            inference_ok = (out is not None and len(out[0]) == len(classes))
        except Exception as exc:
            inference_error = str(exc)

    if inference_ok:
        model_status = "ready"
        http_status = 200
    elif model_file_exists:
        model_status = "error"
        http_status = 503
    else:
        model_status = "unavailable"
        http_status = 503

    payload = {
        "status": "healthy" if inference_ok else "degraded",
        "service": "PoultryScan AI Backend",
        "model_status": model_status,
        "model_file_exists": model_file_exists,
        "inference_verified": inference_ok,
        "classes": classes,
        "num_classes": len(classes),
        "database_connected": os.path.exists(DB_PATH),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if inference_error:
        payload["inference_error"] = inference_error
    return jsonify(payload), http_status


@app.route("/api/model-status")
def api_model_status():
    """Lightweight JSON endpoint polled by the sidebar for live model status."""
    model = app.config.get("MODEL")
    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)

    # Load the verified evaluation summary for the accuracy figure
    eval_accuracy = None
    eval_samples = None
    eval_date = None
    if os.path.exists(EVAL_SUMMARY_PATH):
        try:
            with open(EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
                ev = json.load(f)
            eval_accuracy = ev.get("overall_accuracy")
            eval_samples = ev.get("total_test_samples")
            eval_date = ev.get("evaluation_date")
        except Exception:
            pass

    inference_ok = False
    if model is not None:
        try:
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            out = model.predict(dummy, verbose=0)
            inference_ok = (out is not None and len(out[0]) == len(classes))
        except Exception:
            pass

    if inference_ok:
        status = "ready"
        label = f"Active · {len(classes)} Classes"
    elif model is not None:
        status = "error"
        label = "Inference Error"
    else:
        status = "unavailable"
        label = "Model Unavailable"

    return jsonify({
        "status": status,
        "label": label,
        "num_classes": len(classes),
        "eval_accuracy": eval_accuracy,
        "eval_samples": eval_samples,
        "eval_date": eval_date,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for real-time model predictions on uploaded images."""
    if app.config.get("MODEL") is None:
        return jsonify({"success": False, "error": "Model is not loaded on server."}), 503

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' file included in request."}), 400

    uploaded_file = request.files["image"]
    is_valid, error_msg = validate_image_file(uploaded_file)
    if not is_valid:
        return jsonify({"success": False, "error": error_msg}), 400

    image_type = request.form.get("image_type", "full_body").strip().lower()
    if image_type not in VALID_IMAGE_TYPES:
        image_type = "full_body"

    try:
        saved_filename, file_path = save_secure_upload(uploaded_file)
        
        # Validate suitable poultry image before inference
        model = app.config.get("MODEL")
        is_poultry, validation_error, val_details = validate_poultry_image(
            file_path, image_type=image_type, disease_model=model
        )
        if not is_poultry:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            return jsonify({
                "success": False,
                "error": validation_error,
                "is_poultry": False,
                "validation_details": val_details,
            }), 400

        label, confidence, prob_dict, is_uncertain = predict_image(file_path, image_type=image_type)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Processing error: {str(exc)}"}), 500

    info = DISEASE_INFO.get(label, DISEASE_INFO.get("Healthy", {}))

    return jsonify({
        "success": True,
        "prediction": label,
        "confidence": confidence,
        "is_uncertain": is_uncertain,
        "probabilities": prob_dict,
        "image_type": image_type,
        "image_url": url_for("uploaded_file", filename=saved_filename),
        "disease_info": {
            "name": info.get("name", label),
            "severity": info.get("severity", ""),
            "summary": info.get("summary", ""),
            "recommendation": info.get("recommendation", "")
        }
    })


@app.route("/api/flock-predict", methods=["POST"])
def api_flock_predict():
    """REST API endpoint for multi-chicken flock detection and classification."""
    model = app.config.get("MODEL")
    if model is None:
        return jsonify({"success": False, "error": "Model is not loaded on server."}), 503

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' file included in request."}), 400

    uploaded_file = request.files["image"]
    is_valid, error_msg = validate_image_file(uploaded_file)
    if not is_valid:
        return jsonify({"success": False, "error": error_msg}), 400

    try:
        saved_filename, file_path = save_secure_upload(uploaded_file)
        classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)
        flock_res = analyze_flock_image(file_path, model, classes, upload_dir=UPLOAD_DIR)

        if not flock_res.get("success"):
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                "success": False,
                "error": flock_res.get(
                    "error",
                    "Invalid specimen image — no chicken detected. Please upload a clear chicken photograph."
                ),
                "total_chickens": 0,
            }), 400

        return jsonify({
            "success": True,
            "total_chickens": flock_res["total_chickens"],
            "healthy_count": flock_res["healthy_count"],
            "diseased_count": flock_res["diseased_count"],
            "uncertain_count": flock_res["uncertain_count"],
            "chickens": flock_res["chickens"],
            "annotated_url": url_for("uploaded_file", filename=flock_res["annotated_filename"]),
            "veterinary_disclaimer": flock_res["veterinary_disclaimer"],
        })
    except Exception as exc:
        return jsonify({"success": False, "error": f"Processing error: {str(exc)}"}), 500


@app.route("/api/stats")
def api_stats():
    """Live API stats polled by dashboard charts.

    Normalizes any stale label variants from the DB before aggregating.
    """
    db = get_db()
    classes = app.config.get("CLASS_NAMES", CURRENT_CLASSES)
    rows = db.execute("SELECT disease, COUNT(*) c FROM predictions GROUP BY disease").fetchall()
    counts = {c: 0 for c in classes}
    for r in rows:
        canonical = normalize_label(r["disease"])
        if canonical in counts:
            counts[canonical] += r["c"]
        # Stale labels not in the canonical set are silently dropped from counts
        # (they were already normalized on write; this handles very old records).

    total = sum(counts.values())
    avg = db.execute("SELECT AVG(confidence) a FROM predictions").fetchone()["a"] or 0
    last_id = db.execute("SELECT MAX(id) m FROM predictions").fetchone()["m"] or 0

    return jsonify({
        "counts": counts,
        "total": total,
        "healthy": counts.get("Healthy", 0),
        "diseased": total - counts.get("Healthy", 0),
        "avg": round(avg, 1),
        "last_id": last_id,
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Serve uploaded images securely from uploads/ folder."""
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/research/gradcam/<path:filename>")
def serve_gradcam(filename):
    """Serve Grad-CAM visualization artifacts securely."""
    return send_from_directory(GRADCAM_DIR, filename)


# Error Handlers
@app.errorhandler(413)
def request_entity_too_large(_error):
    flash("Uploaded file exceeds the maximum size limit (16 MB). Please upload a smaller image.", "error")
    return redirect(url_for("upload"))


@app.errorhandler(404)
def page_not_found(_error):
    flash("The requested page could not be found.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Default port: 7860 for HF Spaces, 5000 for local dev.
    # Set debug=False for production safety; use_reloader=False prevents
    # double-process issues when running directly with python app.py.
    port = int(os.environ.get("PORT", 7860))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    app.run(debug=debug_mode, host="0.0.0.0", port=port, use_reloader=False)
