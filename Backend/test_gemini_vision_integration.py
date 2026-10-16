"""
LeafLine AI - Comprehensive Integration Test Suite
--------------------------------------------------
Verifies Gemini Vision-based plant disease detection, image validation (5 MB limit),
non-plant rejection, verified treatment retrieval, and edge case resilience.
"""

import os
import io
import sys
import unittest
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from Backend.gemini_disease import (
    analyze_plant_disease,
    validate_and_preprocess_image,
    MAX_IMAGE_SIZE_BYTES
)
from Backend.treatment_store import (
    find_verified_treatment,
    get_safe_fallback_treatment
)
from Backend.classifier_bridge import (
    reconcile_predictions,
    SUPPORTED_CROPS
)
import Backend.main as main_app


class TestLeafLineGeminiVision(unittest.TestCase):

    def setUp(self):
        self.client = main_app.app.test_client()

    # 1. Invalid Image File
    def test_invalid_image_file(self):
        corrupt_data = io.BytesIO(b"Not an actual image payload")
        resp = self.client.post(
            "/analyze-disease",
            data={"image": (corrupt_data, "bad_image.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error_type"], "validation_error")
        self.assertIn("Malformed or unreadable", data["detail"])

    # 2. Image Larger than 5 MB
    def test_image_larger_than_5mb(self):
        huge_data = io.BytesIO(b"X" * (6 * 1024 * 1024))
        resp = self.client.post(
            "/analyze-disease",
            data={"image": (huge_data, "huge_photo.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error_type"], "file_too_large")
        self.assertIn("5 MB limit", data["detail"])

    # 3. Missing Image
    def test_missing_image(self):
        resp = self.client.post("/analyze-disease", data={})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error_type"], "missing_file")

    # 4. Unrelated Non-Plant Image (Grey Box / Lamp Object)
    def test_unrelated_non_plant_image(self):
        img = Image.new("RGB", (120, 120), color="slategray")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = self.client.post(
            "/analyze-disease",
            data={"image": (buf, "non_plant.jpg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["is_plant_visible"])
        self.assertIn("No plant leaf was detected", data["detail"])

    # 5. Healthy Leaf Image
    def test_healthy_leaf_image(self):
        # Synthetic green leaf image
        img = Image.new("RGB", (200, 200), color=(34, 139, 34))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        res = analyze_plant_disease(buf.getvalue())
        self.assertTrue(res["success"])
        # Reconcile with verified treatment store
        treatment = find_verified_treatment("Tomato", "Healthy")
        self.assertTrue(treatment["is_verified"])
        self.assertTrue(treatment["is_healthy"])
        self.assertEqual(len(treatment["treatment"]), 0)  # Zero chemical sprays for healthy plant

    # 6. Disease with Verified Treatment Record (Tomato Early Blight)
    def test_disease_with_verified_treatment(self):
        treatment = find_verified_treatment("Tomato", "Early Blight")
        self.assertTrue(treatment["is_verified"])
        self.assertEqual(treatment["canonical_label"], "Tomato___Early_blight")
        self.assertGreater(len(treatment["treatment"]), 0)
        self.assertGreater(len(treatment["source_references"]), 0)
        # Check source has URL
        first_source = treatment["source_references"][0]
        self.assertTrue("url" in first_source and first_source["url"].startswith("http"))
        self.assertIn("Mancozeb", " ".join(treatment["treatment"]))

    # 7. Disease with NO Verified Treatment Record (Safe Fallback)
    def test_disease_with_no_verified_treatment(self):
        treatment = find_verified_treatment("Tomato", "Unprecedented Martian Fungus")
        self.assertFalse(treatment["is_verified"])
        self.assertEqual(len(treatment["treatment"]), 0)  # Strictly NO invented chemicals
        self.assertEqual(len(treatment["active_ingredients"]), 0)
        self.assertIn("Do NOT apply synthetic chemical pesticides", treatment["safety_precautions"][0])

    # 8. Unsupported Crop Handling
    def test_unsupported_crop_handling(self):
        consensus = reconcile_predictions(
            gemini_data={"crop": "Avocado", "disease_name": "Root Rot", "uncertainty": "Visual check only"},
            local_pred=None
        )
        self.assertFalse(consensus["is_crop_supported"])
        self.assertIn("not currently in LeafLine's verified regional agronomy package", consensus["crop_support_note"])

    # 9. Conflicting Model Predictions
    def test_conflicting_model_predictions(self):
        gemini_data = {"crop": "Tomato", "disease_name": "Early Blight", "uncertainty": "Visual spots detected"}
        local_pred = {"predicted_label": "Tomato___Septoria_leaf_spot", "confidence": 0.82}
        consensus = reconcile_predictions(gemini_data, local_pred)
        self.assertTrue(consensus["disagreement"])
        self.assertIn("High diagnostic uncertainty", consensus["uncertainty_status"])

    # 10. Missing API Credentials Handling
    def test_missing_api_credentials(self):
        old_key = os.environ.get("GEMINI_API_KEY")
        try:
            os.environ["GEMINI_API_KEY"] = ""
            res = analyze_plant_disease(b"dummy_bytes")
            self.assertFalse(res["success"])
            self.assertEqual(res["error_type"], "config_error")
            self.assertIn("API key is not configured", res["message"])
        finally:
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key

    # 11. Supported Crops List
    def test_supported_crops_list(self):
        self.assertIn("tomato", SUPPORTED_CROPS)
        self.assertIn("chilli", SUPPORTED_CROPS)
        self.assertIn("brinjal", SUPPORTED_CROPS)
        self.assertIn("okra", SUPPORTED_CROPS)
        self.assertIn("cucurbits", SUPPORTED_CROPS)


if __name__ == "__main__":
    unittest.main()
