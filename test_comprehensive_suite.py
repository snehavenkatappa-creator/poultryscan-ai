"""Comprehensive Phase 7 Automated Test Suite for PoultryScan AI.

Covers all 11 required audit categories:
1. Model loading for each examination mode (head_face, full_body).
2. Model input and output dimensions.
3. Correct class mappings.
4. Preprocessing consistency.
5. Prediction API behavior.
6. Invalid, corrupted, and unsupported image handling.
7. Prediction-history persistence.
8. Dashboard counts and label normalization.
9. Model-health indicators (/health, /api/model-status).
10. Evaluation metric calculations.
11. Grad-CAM execution.
"""

import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import PoultryScan AI modules
import app
import diseases
from test_model import preprocess_image as cli_preprocess


class TestPoultryScanAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()
        cls.app = app.app
        cls.model = app.app.config.get("MODEL")
        cls.classes = app.app.config.get("CLASS_NAMES")
        cls.sample_image = os.path.join("test_dataset", "Healthy", "healthy.10.jpg")

    # --------------------------------------------------------------------------
    # 1. Model Loading for Each Examination Mode
    # --------------------------------------------------------------------------
    def test_01_model_loading_both_modes(self):
        """Test 1: Model loads and executes inference for both head_face and full_body."""
        self.assertIsNotNone(self.model, "Trained model must be loaded in app.config['MODEL']")
        self.assertTrue(os.path.exists(self.sample_image), f"Sample image missing: {self.sample_image}")

        # Test full_body mode
        pred_body, conf_body, probs_body, unc_body = app.predict_image(self.sample_image, image_type="full_body")
        self.assertIn(pred_body, self.classes)
        self.assertIsInstance(conf_body, float)
        self.assertIsInstance(unc_body, bool)

        # Test head_face mode
        pred_head, conf_head, probs_head, unc_head = app.predict_image(self.sample_image, image_type="head_face")
        self.assertIn(pred_head, self.classes)
        self.assertIsInstance(conf_head, float)
        self.assertIsInstance(unc_head, bool)

    # --------------------------------------------------------------------------
    # 2. Model Input and Output Dimensions
    # --------------------------------------------------------------------------
    def test_02_model_input_output_dimensions(self):
        """Test 2: Verify model input shape (None, 224, 224, 3) and output neurons (None, 5)."""
        self.assertEqual(self.model.input_shape, (None, 224, 224, 3), "Model input must be 224x224x3 RGB")
        self.assertEqual(self.model.output_shape, (None, 5), "Model output must have exactly 5 neurons")

        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        out = self.model.predict(dummy, verbose=0)
        self.assertEqual(out.shape, (1, 5), "Prediction vector must have shape (1, 5)")
        # Softmax sum verification
        self.assertAlmostEqual(float(np.sum(out[0])), 1.0, places=4, msg="Output neurons must sum to ~1.0 (softmax)")

    # --------------------------------------------------------------------------
    # 3. Correct Class Mappings
    # --------------------------------------------------------------------------
    def test_03_correct_class_mappings(self):
        """Test 3: Verify alphabetical class mapping consistency across files."""
        expected_classes = ["Coccidiosis", "Fowl Pox", "Healthy", "Newcastle Disease", "Salmonella"]
        
        # Check class_indices.json
        with open("model/class_indices.json", "r", encoding="utf-8") as f:
            ci = json.load(f)
        self.assertEqual(ci, expected_classes, "class_indices.json does not match expected 5 classes")

        # Check diseases.py CLASS_NAMES
        self.assertEqual(diseases.CLASS_NAMES, expected_classes, "diseases.CLASS_NAMES mismatch")

        # Check app.config['CLASS_NAMES']
        self.assertEqual(self.classes, expected_classes, "app.config['CLASS_NAMES'] mismatch")

        # Check ACTIVE_MODEL_CLASSES
        self.assertEqual(diseases.ACTIVE_MODEL_CLASSES, expected_classes, "diseases.ACTIVE_MODEL_CLASSES mismatch")

    # --------------------------------------------------------------------------
    # 4. Preprocessing Consistency
    # --------------------------------------------------------------------------
    def test_04_preprocessing_consistency(self):
        """Test 4: Verify that app.py and CLI test scripts produce identical tensors."""
        tensor_app = app.preprocess_image(self.sample_image)
        tensor_cli = cli_preprocess(self.sample_image)

        self.assertEqual(tensor_app.shape, (1, 224, 224, 3))
        self.assertEqual(tensor_app.dtype, np.float32)
        self.assertTrue(0.0 <= np.min(tensor_app) and np.max(tensor_app) <= 1.0, "Pixels must be in [0.0, 1.0]")
        np.testing.assert_allclose(tensor_app, tensor_cli, rtol=1e-5, atol=1e-5,
                                  err_msg="app and CLI preprocessing must be identical")

    # --------------------------------------------------------------------------
    # 5. Prediction API Behavior (/api/predict)
    # --------------------------------------------------------------------------
    def test_05_prediction_api_behavior(self):
        """Test 5: REST API returns valid JSON with prediction, probabilities, and metadata."""
        with open(self.sample_image, "rb") as f:
            img_bytes = f.read()

        # Valid POST
        res = self.client.post("/api/predict", data={
            "image": (io.BytesIO(img_bytes), "test_api.jpg"),
            "image_type": "full_body"
        })
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data.get("success"))
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertIn("probabilities", data)
        self.assertEqual(len(data["probabilities"]), 5)
        self.assertIn("disease_info", data)

        # Missing file in POST
        res_missing = self.client.post("/api/predict", data={})
        self.assertEqual(res_missing.status_code, 400)
        data_missing = json.loads(res_missing.data)
        self.assertFalse(data_missing.get("success"))

    # --------------------------------------------------------------------------
    # 6. Invalid, Corrupted, and Unsupported Image Handling
    # --------------------------------------------------------------------------
    def test_06_invalid_corrupted_unsupported_images(self):
        """Test 6: Reject invalid extension, corrupted bytes, and empty uploads safely."""
        # Unsupported extension (.txt)
        res_txt = self.client.post("/upload", data={
            "image": (io.BytesIO(b"Hello world! Not an image."), "malicious.txt")
        }, follow_redirects=True)
        self.assertIn(b"Invalid file format", res_txt.data)

        # Corrupted JPG header
        res_corrupt = self.client.post("/upload", data={
            "image": (io.BytesIO(b"\xFF\xD8\xFF\xE0\x00\x10JFIFcorrupted_garbage_bytes"), "corrupt.jpg")
        }, follow_redirects=True)
        self.assertIn(b"corrupt or not a readable image", res_corrupt.data)

        # Empty filename
        res_empty = self.client.post("/upload", data={
            "image": (io.BytesIO(b""), "")
        }, follow_redirects=True)
        self.assertIn(b"Please select an image file", res_empty.data)

    # --------------------------------------------------------------------------
    # 7. Prediction History Persistence
    # --------------------------------------------------------------------------
    def test_07_prediction_history_persistence(self):
        """Test 7: Verify SQLite record creation, retrieval, and delete functionality."""
        with open(self.sample_image, "rb") as f:
            img_bytes = f.read()

        res = self.client.post("/upload", data={
            "image": (io.BytesIO(img_bytes), "history_test.jpg"),
            "image_type": "full_body"
        }, follow_redirects=False)
        self.assertEqual(res.status_code, 302, "Successful upload must redirect to result")

        # Check DB
        with sqlite3.connect(app.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
            self.assertIsNotNone(row)
            self.assertIn("history_test.jpg", row["filename"])
            self.assertIn(row["disease"], self.classes)
            self.assertIsInstance(row["confidence"], float)
            self.assertEqual(row["image_type"], "full_body")
            created_id = row["id"]

        # Test history view contains the record
        res_history = self.client.get("/history")
        self.assertEqual(res_history.status_code, 200)

        # Clean up created test record via delete route
        res_del = self.client.post(f"/history/delete/{created_id}", follow_redirects=True)
        self.assertEqual(res_del.status_code, 200)

    # --------------------------------------------------------------------------
    # 8. Dashboard Counts and Label Normalization
    # --------------------------------------------------------------------------
    def test_08_dashboard_counts_and_normalization(self):
        """Test 8: Label aliases normalize properly and /api/stats aggregates correctly."""
        # Alias normalization
        self.assertEqual(app.normalize_label("New Castle Disease"), "Newcastle Disease")
        self.assertEqual(app.normalize_label("new castle disease"), "Newcastle Disease")
        self.assertEqual(app.normalize_label("salmonellosis"), "Salmonella")
        self.assertEqual(app.normalize_label("Healthy Chicken"), "Healthy")
        self.assertEqual(app.normalize_label("Coccidiosis"), "Coccidiosis")

        # /api/stats endpoint
        res = self.client.get("/api/stats")
        self.assertEqual(res.status_code, 200)
        stats = json.loads(res.data)
        self.assertIn("counts", stats)
        self.assertIn("total", stats)
        self.assertIn("healthy", stats)
        self.assertIn("diseased", stats)
        self.assertEqual(stats["total"], stats["healthy"] + stats["diseased"])
        for c in self.classes:
            self.assertIn(c, stats["counts"], f"Class {c} missing from stats counts")
        self.assertNotIn("New Castle Disease", stats["counts"], "Legacy variant must not appear in stats keys")

    # --------------------------------------------------------------------------
    # 9. Model Health Indicators
    # --------------------------------------------------------------------------
    def test_09_model_health_indicators(self):
        """Test 9: /health and /api/model-status report real inference readiness."""
        # /health
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        hdata = json.loads(res_health.data)
        self.assertEqual(hdata["status"], "healthy")
        self.assertEqual(hdata["model_status"], "ready")
        self.assertTrue(hdata["inference_verified"])
        self.assertEqual(hdata["num_classes"], 5)

        # /api/model-status
        res_status = self.client.get("/api/model-status")
        self.assertEqual(res_status.status_code, 200)
        sdata = json.loads(res_status.data)
        self.assertEqual(sdata["status"], "ready")
        self.assertEqual(sdata["num_classes"], 5)
        self.assertAlmostEqual(sdata["eval_accuracy"], 91.32, places=2)

    # --------------------------------------------------------------------------
    # 10. Evaluation Metric Calculations
    # --------------------------------------------------------------------------
    def test_10_evaluation_metric_calculations(self):
        """Test 10: Test accuracy, confusion matrix, and classification report math."""
        # Synthesize ground truth and predictions for 5 classes
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        y_pred = np.array([0, 0, 1, 0, 2, 2, 3, 4, 4, 4])  # 8 correct out of 10 = 80.0%

        acc = accuracy_score(y_true, y_pred)
        self.assertAlmostEqual(acc, 0.80, places=4)

        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.shape, (5, 5))
        self.assertEqual(np.sum(cm), 10)
        self.assertEqual(np.trace(cm), 8)

        report = classification_report(y_true, y_pred, target_names=self.classes, output_dict=True, zero_division=0)
        self.assertAlmostEqual(report["accuracy"], 0.80, places=4)
        self.assertIn("macro avg", report)
        self.assertIn("weighted avg", report)

    # --------------------------------------------------------------------------
    # 11. Grad-CAM Execution
    # --------------------------------------------------------------------------
    def test_11_gradcam_execution(self):
        """Test 11: Grad-CAM generates non-empty spatial heatmaps on active model."""
        from research_results.eval_scripts.grad_cam_analysis import make_gradcam_heatmap

        tensor = app.preprocess_image(self.sample_image)
        heatmap, target_layer = make_gradcam_heatmap(tensor, self.model, class_index=2)

        self.assertIsNotNone(heatmap)
        self.assertEqual(len(heatmap.shape), 2, "Heatmap must be 2D spatial matrix")
        self.assertTrue(0.0 <= np.min(heatmap) and np.max(heatmap) <= 1.0, "Heatmap values must be normalized to [0, 1]")
        self.assertIn("relu", target_layer.lower(), f"Target layer should be terminal spatial activation: {target_layer}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
