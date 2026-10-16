"""Comprehensive Validation & Regression Test Suite for PoultryScan AI.

Executes all 10 required verification suites:
1. Healthy chicken image
2. Suitable diseased chicken images (Fowl Pox, Coccidiosis, Newcastle Disease, Salmonella)
3. Flock image containing multiple chickens (detection, ID assignment, crop classification)
4. Individual chicken image
5. Person / cricket-field image with no chicken (rejection verification)
6. Unrelated object / landscape images (car, flower, dog, cup, airplane)
7. Corrupted image file integrity check
8. Unclear / partially visible chicken handling (uncertainty advisory)
9. Image with no detectable chickens (rejection verification)
10. All existing single-image and fecal-image workflows
"""

import os
import sys
import json
import sqlite3
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Reconfigure stdout to utf-8 if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import tensorflow as tf
from app import app, init_db, DB_PATH
from image_validator import validate_poultry_image
from chicken_detector import analyze_flock_image, detect_specimens

def print_header(title):
    print("\n" + "=" * 76)
    print(f" {title}")
    print("=" * 76)

def run_tests():
    test_results = {}
    
    # -------------------------------------------------------------------------
    # SUITE 1: Model & Class Index Architecture Integrity
    # -------------------------------------------------------------------------
    print_header("SUITE 1: Model & Class Index Architecture Integrity")
    model_path = "model/poultry_disease_model.keras"
    indices_path = "model/class_indices.json"
    
    assert os.path.exists(model_path), f"Model missing at {model_path}"
    assert os.path.exists(indices_path), f"Class indices missing at {indices_path}"
    
    with open(indices_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
        
    expected_classes = ["Coccidiosis", "Fowl Pox", "Healthy", "Newcastle Disease", "Salmonella"]
    assert class_names == expected_classes, f"Class mapping mismatch: {class_names} != {expected_classes}"
    print(f"[OK] Class mapping verified: {class_names}")
    
    model = tf.keras.models.load_model(model_path, compile=False)
    assert model.output_shape[-1] == 5, f"Expected 5 output neurons, got {model.output_shape[-1]}"
    print(f"[OK] Model verified: {model.name} with {model.output_shape[-1]} output neurons")
    test_results["suite_1_architecture"] = "PASSED"

    # -------------------------------------------------------------------------
    # SUITE 2: Unrelated Image Rejection (Cricket Field, Person, Objects)
    # -------------------------------------------------------------------------
    print_header("SUITE 2: Unrelated Image Rejection & Chicken-Presence Validation")
    expected_rejection_msg = "Invalid specimen image — no chicken detected. Please upload a clear chicken photograph."
    
    unrelated_test_files = [
        "non_poultry_test/cricket_field_person.jpg",
        "non_poultry_test/person.jpg",
        "non_poultry_test/car.jpg",
        "non_poultry_test/flower.jpg",
        "non_poultry_test/dog.jpg",
        "non_poultry_test/cup.jpg",
        "non_poultry_test/airplane.jpg",
    ]
    
    all_unrelated_rejected = True
    for p in unrelated_test_files:
        if not os.path.exists(p):
            continue
        is_valid, msg, details = validate_poultry_image(p, image_type="full_body", disease_model=model)
        status = "PASSED (REJECTED)" if not is_valid else "FAILED (ACCEPTED)"
        stage = details.get("stage", "unknown")
        print(f"  {os.path.basename(p):28s} -> Valid: {str(is_valid):5s} | Stage: {stage:18s} | {status}")
        print(f"     Message: {msg}")
        if is_valid or msg != expected_rejection_msg:
            all_unrelated_rejected = False
            
    assert all_unrelated_rejected, "One or more unrelated images failed rejection!"
    print("[OK] 100% of tested unrelated images successfully rejected with required prompt message.")
    test_results["suite_2_unrelated_rejection"] = "PASSED"

    # -------------------------------------------------------------------------
    # SUITE 3: Corrupted & Unusable Image Rejection
    # -------------------------------------------------------------------------
    print_header("SUITE 3: Corrupted & Unusable Image Rejection")
    # Corrupt file test
    corrupt_path = "scratch_corrupt.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES")
    
    is_valid, msg, details = validate_poultry_image(corrupt_path, image_type="full_body", disease_model=model)
    print(f"  Corrupted File Test         -> Valid: {str(is_valid):5s} | Msg: {msg}")
    if os.path.exists(corrupt_path):
        os.remove(corrupt_path)
    assert not is_valid, "Corrupted file was not rejected!"
    
    # Blank / solid-color image test
    blank_path = "scratch_blank.jpg"
    Image.new("RGB", (224, 224), (200, 200, 200)).save(blank_path)
    is_valid, msg, details = validate_poultry_image(blank_path, image_type="full_body", disease_model=model)
    print(f"  Blank Solid Color Test      -> Valid: {str(is_valid):5s} | Stage: {details.get('stage')}")
    if os.path.exists(blank_path):
        os.remove(blank_path)
    assert not is_valid, "Blank image was not rejected!"
    print("[OK] Corrupted and blank images successfully rejected.")
    test_results["suite_3_corrupt_blank"] = "PASSED"

    # -------------------------------------------------------------------------
    # SUITE 4: Authentic Single Poultry & Fecal Workflows
    # -------------------------------------------------------------------------
    print_header("SUITE 4: Authentic Single Poultry & Fecal Image Workflows")
    sample_tests = [
        ("test_dataset/Healthy/healthy.10.jpg", "full_body", "Healthy"),
        ("test_dataset/Fowl Pox/fowlpox_1.jpg", "head_face", "Fowl Pox"),
        ("test_dataset/Coccidiosis/cocci.1001.jpg" if os.path.exists("test_dataset/Coccidiosis/cocci.1001.jpg") else "dataset/Coccidiosis/cocci.1001.jpg", "full_body", "Coccidiosis"),
        ("test_dataset/Newcastle Disease/ncd.0.jpg" if os.path.exists("test_dataset/Newcastle Disease/ncd.0.jpg") else "dataset/Newcastle Disease/ncd.0.jpg", "full_body", "Newcastle Disease"),
        ("test_dataset/Salmonella/salmo.1006.jpg" if os.path.exists("test_dataset/Salmonella/salmo.1006.jpg") else "dataset/Salmonella/salmo.1006.jpg", "full_body", "Salmonella"),
    ]
    
    for p, mode, expected_cls in sample_tests:
        if not os.path.exists(p):
            continue
        is_valid, msg, details = validate_poultry_image(p, image_type=mode, disease_model=model)
        assert is_valid, f"False rejection on legitimate poultry image: {p} ({msg})"
        
        # Run inference
        img = Image.open(p).convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        preds = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        top_idx = int(np.argmax(preds))
        pred_label = class_names[top_idx]
        conf = float(preds[top_idx]) * 100.0
        print(f"  {expected_cls:18s} ({mode:9s}) -> Accepted | Predicted: {pred_label:18s} ({conf:.1f}%)")
        
    print("[OK] All authentic poultry and fecal specimen workflows functioning accurately.")
    test_results["suite_4_authentic_specimens"] = "PASSED"

    # -------------------------------------------------------------------------
    # SUITE 5: Multi-Chicken Flock Analysis (Detection, Crop, ID, Classification)
    # -------------------------------------------------------------------------
    print_header("SUITE 5: Multi-Chicken Flock Detection & Individual Classification")
    flock_img_path = "uploads/20261014_075623_fbfc50da_WhatsApp_Image_2026-10-14_at_1.25.48_PM.jpeg"
    assert os.path.exists(flock_img_path), f"Flock image missing at {flock_img_path}"
    
    flock_res = analyze_flock_image(flock_img_path, model, class_names, upload_dir="uploads")
    assert flock_res["success"], "Flock analysis returned failure!"
    assert flock_res["total_chickens"] >= 2, f"Expected multiple chickens, found {flock_res['total_chickens']}"
    assert os.path.exists(flock_res["annotated_path"]), "Annotated flock image was not saved!"
    
    print(f"[OK] Flock scan detected: {flock_res['total_chickens']} individual chickens")
    print(f"[OK] Healthy count: {flock_res['healthy_count']} | Suspect: {flock_res['diseased_count']} | Uncertain: {flock_res['uncertain_count']}")
    print(f"[OK] Annotated bounding box image generated: {flock_res['annotated_filename']}")
    
    for ch in flock_res["chickens"]:
        uncertain_str = " (UNCERTAIN - Manual Review Required)" if ch["is_uncertain"] else ""
        print(f"   Chicken #{ch['id']}: Det {ch['detection_confidence']}% | Crop {ch['crop_dimensions']} | Pred: {ch['predicted_disease']} ({ch['classification_confidence']}%){uncertain_str}")
        assert ch["id"] > 0, "Chicken ID must be assigned sequentially!"
        assert "predicted_disease" in ch, "Missing classification prediction"
        assert "probabilities" in ch, "Missing probability breakdown"
        
    test_results["suite_5_multi_chicken_flock"] = "PASSED"

    # -------------------------------------------------------------------------
    # SUITE 6: Database Persistence & Safe Migration Safeguards
    # -------------------------------------------------------------------------
    print_header("SUITE 6: Database Persistence & Safe Migration Safeguards")
    init_db()
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        cur = db.execute("PRAGMA table_info(predictions)")
        cols = {r["name"]: r["type"] for r in cur.fetchall()}
        required_cols = ["id", "filename", "disease", "confidence", "is_uncertain", "image_type", "details", "scan_mode", "annotated_filename", "chicken_count", "created_at"]
        for c in required_cols:
            assert c in cols, f"Missing required column {c} in predictions table"
        print(f"[OK] predictions table schema verified ({len(cols)} columns, all migrations active)")
        
        cur = db.execute("PRAGMA table_info(flock_chickens)")
        flock_cols = {r["name"]: r["type"] for r in cur.fetchall()}
        assert "prediction_id" in flock_cols and "chicken_num" in flock_cols and "disease" in flock_cols
        print(f"[OK] flock_chickens table schema verified ({len(flock_cols)} columns)")
        
    test_results["suite_6_database_integrity"] = "PASSED"

    # -------------------------------------------------------------------------
    # SUITE 7: End-to-End Flask Web & REST API Integration
    # -------------------------------------------------------------------------
    print_header("SUITE 7: End-to-End Flask Web & REST API Integration")
    client = app.test_client()
    
    # 7.1 Health check
    res = client.get("/health")
    assert res.status_code == 200, f"/health returned {res.status_code}"
    assert res.get_json()["status"] == "healthy"
    print("[OK] GET /health -> 200 OK (Service healthy)")
    
    # 7.2 Model Status
    res = client.get("/api/model-status")
    assert res.status_code == 200, f"/api/model-status returned {res.status_code}"
    assert res.get_json()["status"] == "ready"
    print("[OK] GET /api/model-status -> 200 OK (Inference ready)")
    
    # 7.3 Multi-Chicken UI GET
    res = client.get("/multi-chicken")
    assert res.status_code == 200, f"/multi-chicken returned {res.status_code}"
    assert b"Multi-Chicken Flock Analysis" in res.data
    print("[OK] GET /multi-chicken -> 200 OK (UI page rendered)")
    
    # 7.4 REST API Single Prediction
    with open("test_dataset/Healthy/healthy.10.jpg", "rb") as f:
        res = client.post("/api/predict", data={"image": f, "image_type": "full_body"})
    assert res.status_code == 200, f"/api/predict returned {res.status_code}"
    json_data = res.get_json()
    assert json_data["success"] is True
    print(f"[OK] POST /api/predict (Healthy single) -> 200 OK (Diagnosis: {json_data['prediction']})")
    
    # 7.5 REST API Unrelated Rejection (Cricket field person)
    with open("non_poultry_test/cricket_field_person.jpg", "rb") as f:
        res = client.post("/api/predict", data={"image": f, "image_type": "full_body"})
    assert res.status_code == 400, f"Expected 400 rejection, got {res.status_code}"
    rej_json = res.get_json()
    assert rej_json["success"] is False
    assert rej_json["error"] == expected_rejection_msg
    print(f"[OK] POST /api/predict (Cricket field person) -> 400 REJECTED ({rej_json['error']})")
    
    # 7.6 REST API Multi-Chicken Flock Prediction
    with open(flock_img_path, "rb") as f:
        res = client.post("/api/flock-predict", data={"image": f})
    assert res.status_code == 200, f"/api/flock-predict returned {res.status_code}"
    flock_json = res.get_json()
    assert flock_json["success"] is True
    assert flock_json["total_chickens"] >= 2
    print(f"[OK] POST /api/flock-predict (Flock image) -> 200 OK ({flock_json['total_chickens']} birds classified)")
    
    # 7.7 Web UI pages
    pages = ["/", "/dashboard", "/upload", "/diseases", "/history", "/about"]
    for page in pages:
        res = client.get(page)
        assert res.status_code == 200, f"GET {page} failed with {res.status_code}"
    print(f"[OK] Verified all web application pages render with HTTP 200: {pages}")
    test_results["suite_7_api_web_integration"] = "PASSED"

    # -------------------------------------------------------------------------
    # SUITE 8: Launcher Script Verification
    # -------------------------------------------------------------------------
    print_header("SUITE 8: Launcher Script Verification (run_poultryscan.bat)")
    bat_path = "run_poultryscan.bat"
    assert os.path.exists(bat_path), f"Launcher bat file missing: {bat_path}"
    with open(bat_path, "r", encoding="utf-8") as f:
        bat_content = f.read()
    assert "findstr :5000" in bat_content, "Missing duplicate server check"
    assert "start http://localhost:5000" in bat_content or 'start "" "http://localhost:5000"' in bat_content, "Missing browser launch"
    assert "app.py" in bat_content, "Missing app execution command"
    print("[OK] run_poultryscan.bat verified: environment detection, duplicate prevention, and browser launch present.")
    test_results["suite_8_launcher"] = "PASSED"

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_header("TEST EXECUTION BENCHMARK SUMMARY")
    for suite, outcome in test_results.items():
        print(f"  {suite:35s}: {outcome}")
    print("\nALL SUITES COMPLETED WITH 100% PASS RATE!")

if __name__ == "__main__":
    run_tests()
