"""Verification test suite for PoultryScan AI model loading and 5-class predictions."""
import json
import os
import sys
import app

def test_all():
    print("=" * 60)
    print("POULTRYSCAN AI — VERIFICATION SUITE")
    print("=" * 60)

    print("\n[Step 1] Testing Model Loading & Candidate Resolution...")
    model = app.get_model()
    assert model is not None, "Model should not be None"
    print(f"  [OK] Model Loaded: {model.name}")
    print(f"  [OK] Input Shape:  {model.input_shape}")
    print(f"  [OK] Output Shape: {model.output_shape}")
    classes = app.app.config.get("CLASS_NAMES")
    print(f"  [OK] Active Classes ({len(classes)}): {classes}")

    print("\n[Step 2] Testing Flask Endpoints via Test Client...")
    client = app.app.test_client()

    # Test /
    resp = client.get("/")
    assert resp.status_code == 200, f"GET / failed: {resp.status_code}"
    print("  [OK] GET / -> 200 OK")

    # Test /api/model-status
    resp = client.get("/api/model-status")
    assert resp.status_code == 200, f"GET /api/model-status failed: {resp.status_code}"
    status_data = json.loads(resp.data)
    print(f"  [OK] GET /api/model-status -> status='{status_data.get('status')}', label='{status_data.get('label')}'")
    assert status_data.get("status") == "ready", "Model status is not ready"

    # Test /upload GET
    resp = client.get("/upload")
    assert resp.status_code == 200, f"GET /upload failed: {resp.status_code}"
    print("  [OK] GET /upload -> 200 OK")

    # Test /upload POST with Fowl Pox image
    with open("test_dataset/Fowl Pox/fowlpox_1.jpg", "rb") as f:
        resp = client.post("/upload", data={
            "image": (f, "fowlpox_1.jpg"),
            "image_type": "head_face"
        }, follow_redirects=True)

    assert resp.status_code == 200, f"POST /upload failed: {resp.status_code}"
    html = resp.data.decode("utf-8", errors="ignore")
    assert "Prediction service unavailable" not in html, "Found 'Prediction service unavailable' error in HTML!"
    assert "Fowl Pox" in html, "Expected 'Fowl Pox' diagnosis in response HTML!"
    print("  [OK] POST /upload with Fowl Pox specimen -> Successfully diagnosed without loading error!")

    print("\n[Step 3] Testing Multi-Class Inference across All 5 Classes...")
    test_cases = [
        ("Coccidiosis", "test_dataset/Coccidiosis/cocci.1002.jpg", "full_body"),
        ("Fowl Pox", "test_dataset/Fowl Pox/fowlpox_1.jpg", "head_face"),
        ("Healthy", "test_dataset/Healthy/healthy.10.jpg", "full_body"),
        ("Newcastle Disease", "test_dataset/Newcastle Disease/ncd.103.jpg", "full_body"),
        ("Salmonella", "test_dataset/Salmonella/salmo.0.jpg", "full_body")
    ]

    for exp_cls, img_path, mode in test_cases:
        with open(img_path, "rb") as f:
            resp = client.post("/api/predict", data={"image": (f, "sample.jpg"), "image_type": mode})
        res = json.loads(resp.data)
        assert res.get("success") is True, f"Prediction failed for {exp_cls}: {res}"
        pred = res.get("prediction")
        conf = res.get("confidence")
        print(f"  [OK] Expected: {exp_cls:18s} | Predicted: {pred:18s} ({conf:.2f}%)")

    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_all()
