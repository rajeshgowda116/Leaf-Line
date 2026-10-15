"""
LeafLine - Automated Verification Test Suite for Accuracy, Routing & Abstention
--------------------------------------------------------------------------------
Validates all requirements specified in Phase 7:
1. Valid healthy leaf images.
2. Valid diseased leaf images.
3. Non-leaf images (rejection).
4. Unsupported crops.
5. Corrupt and oversized files.
6. Model-output and label-count mismatches.
7. Low-confidence predictions (abstention & non-defaulting to Healthy).
8. Crop-label mismatches (Corn not forced to Tomato).
9. Treatment lookup mismatches (ICAR data consistency).
10. Frontend / API response consistency.
"""

import io
import os
import sys
import json
import pytest
import numpy as np
from PIL import Image

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Ensure Backend and Backend/Disease are on sys.path
DISEASE_DIR = os.path.join(BACKEND_DIR, "Disease")
if DISEASE_DIR not in sys.path:
    sys.path.insert(0, DISEASE_DIR)
os.environ["TESTING"] = "1"

from api_backend import app, CLASS_NAMES, model, IMG_SIZE
from leaf_validator import is_plant_leaf, validate_image_file
from treatment_service import get_treatment_info, build_marketplace_links


@pytest.fixture
def flask_client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def make_leaf_bytes(color=(34, 139, 34), size=(128, 128)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_non_leaf_bytes(color=(255, 69, 0), size=(128, 128)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# -------------------------------------------------------------
# 1. Class Order & Index Alignment
# -------------------------------------------------------------
def test_class_names_standard_alphabetical_order():
    """Verify that CLASS_NAMES exactly follows standard ASCII alphabetical sorting."""
    assert len(CLASS_NAMES) == 38
    # Test specific key indices where the old case-insensitive list had severe bugs:
    assert CLASS_NAMES[5] == "Cherry___Powdery_mildew", "Index 5 must be Powdery Mildew, NOT Healthy"
    assert CLASS_NAMES[6] == "Cherry___healthy", "Index 6 must be Cherry Healthy"
    assert CLASS_NAMES[9] == "Corn___Northern_Leaf_Blight", "Index 9 must be Corn Northern Leaf Blight, NOT Healthy"
    assert CLASS_NAMES[10] == "Corn___healthy", "Index 10 must be Corn Healthy"
    assert CLASS_NAMES[21] == "Potato___Late_blight", "Index 21 must be Potato Late Blight, NOT Healthy"
    assert CLASS_NAMES[22] == "Potato___healthy", "Index 22 must be Potato Healthy"
    assert CLASS_NAMES[26] == "Strawberry___Leaf_scorch", "Index 26 must be Strawberry Leaf Scorch, NOT Healthy"
    assert CLASS_NAMES[27] == "Strawberry___healthy", "Index 27 must be Strawberry Healthy"
    assert CLASS_NAMES[30] == "Tomato___Late_blight", "Index 30 must be Tomato Late Blight, NOT Healthy"
    assert CLASS_NAMES[37] == "Tomato___healthy", "Index 37 must be Tomato Healthy"


def test_model_output_and_label_count_match():
    """Verify model output shape matches 38 classes."""
    assert hasattr(model, "output_shape")
    assert model.output_shape[-1] == len(CLASS_NAMES) == 38


# -------------------------------------------------------------
# 2. Non-Leaf Rejection
# -------------------------------------------------------------
def test_non_leaf_rejection_flame_human_suit():
    """Verify non-leaf objects are rejected with required message."""
    # Flame / Diya
    flame_bytes = make_non_leaf_bytes(color=(255, 69, 0))
    is_leaf, conf, msg, meta = is_plant_leaf(Image.open(io.BytesIO(flame_bytes)))
    assert is_leaf is False
    assert "does not appear to contain a plant leaf" in msg

    # Human skin tone
    skin_bytes = make_non_leaf_bytes(color=(235, 185, 155))
    is_leaf, conf, msg, meta = is_plant_leaf(Image.open(io.BytesIO(skin_bytes)))
    assert is_leaf is False

    # Plain white wall
    white_bytes = make_non_leaf_bytes(color=(250, 250, 250))
    is_leaf, conf, msg, meta = is_plant_leaf(Image.open(io.BytesIO(white_bytes)))
    assert is_leaf is False


# -------------------------------------------------------------
# 3. Low-Confidence Abstention & Never Defaulting to Healthy
# -------------------------------------------------------------
def test_low_confidence_abstention_never_reports_healthy(flask_client, monkeypatch):
    """
    Test that when model produces low-confidence / high-entropy output,
    the API marks the status as 'Inconclusive' and NEVER claims the plant is Healthy.
    """
    # Mock model.predict to return near-uniform distribution across all 38 classes (p ~ 0.026)
    uniform_preds = np.ones((1, 38), dtype=np.float32) / 38.0
    monkeypatch.setattr(model, "predict", lambda *args, **kwargs: uniform_preds)

    leaf_bytes = make_leaf_bytes(color=(34, 139, 34))
    data = {"file": (io.BytesIO(leaf_bytes), "ambiguous_leaf.jpg", "image/jpeg")}
    response = flask_client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()

    assert res["is_inconclusive"] is True
    # Crucial specification check: status must NOT default to Healthy!
    assert res["status"] == "Inconclusive"
    assert "inconclusive" in res["message"].lower()


# -------------------------------------------------------------
# 4. Treatment Knowledge Base Consistency
# -------------------------------------------------------------
def test_treatment_lookup_for_corn_northern_leaf_blight():
    """Verify that Corn Northern Leaf Blight returns full ICAR treatment."""
    info = get_treatment_info("Corn___Northern_Leaf_Blight")
    assert info["status_type"] == "supported_disease"
    assert info["crop"] == "Corn"
    assert info["disease"] == "Northern Leaf Blight"
    assert len(info["treatment"]) > 0
    assert len(info["active_ingredients"]) > 0
    assert len(info["marketplace_links"]) > 0


def test_treatment_lookup_healthy_suppresses_chemicals():
    """Verify healthy corn leaf returns NO chemicals and NO product links."""
    info = get_treatment_info("Corn___healthy")
    assert info["status_type"] == "healthy"
    assert info["treatment"] == []
    assert info["active_ingredients"] == []
    assert info["marketplace_links"] == []
    assert "healthy" in info["disclaimer"].lower()


def test_treatment_lookup_inconclusive_suppresses_chemicals():
    """Verify inconclusive query returns NO chemicals and NO product links."""
    info = get_treatment_info("Corn___Northern_Leaf_Blight", is_inconclusive=True)
    assert info["status_type"] == "inconclusive"
    assert info["treatment"] == []
    assert info["active_ingredients"] == []
    assert info["marketplace_links"] == []
    assert "inconclusive" in info["disclaimer"].lower()


# -------------------------------------------------------------
# 5. Crop Routing & Multi-Crop Support in FastAPI
# -------------------------------------------------------------
def test_crop_detection_routing_corn_not_forced_to_tomato():
    """Verify FastAPI backend handles Corn without forcing it to Tomato."""
    from services.prediction_service import analyze_leaf_image
    from services.model_service import CROP_REGISTRY

    assert "corn" in CROP_REGISTRY
    assert "potato" in CROP_REGISTRY
    assert CROP_REGISTRY["corn"]["display_name"] == "Corn (Maize)"

    leaf_bytes = make_leaf_bytes(color=(46, 139, 87))
    resp = analyze_leaf_image(crop="corn", file_bytes=leaf_bytes)
    assert resp.crop == "corn", "Explicit corn selection must stay corn, not forced to tomato"
    assert resp.is_leaf is True


def test_unsupported_crop_handling():
    """Verify unsupported crops raise HTTP 400 with helpful message."""
    from fastapi import HTTPException
    from services.prediction_service import analyze_leaf_image

    leaf_bytes = make_leaf_bytes()
    with pytest.raises(HTTPException) as exc:
        analyze_leaf_image(crop="papaya", file_bytes=leaf_bytes)
    assert exc.value.status_code == 400
    assert "not supported" in exc.value.detail.lower()


# -------------------------------------------------------------
# 6. Corrupt and Oversized Files
# -------------------------------------------------------------
def test_corrupt_and_oversized_file_rejection(flask_client):
    # Corrupt
    data = {"file": (io.BytesIO(b"garbagebytesnotanimage"), "corrupt.jpg", "image/jpeg")}
    resp = flask_client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False

    # Oversized
    is_valid, msg, _ = validate_image_file(make_leaf_bytes(), max_size_bytes=50)
    assert is_valid is False
    assert "exceeds" in msg.lower()


# -------------------------------------------------------------
# 7. Frontend / API Response Consistency
# -------------------------------------------------------------
def test_flask_api_response_schema_consistency(flask_client):
    """Verify all fields expected by frontend/index.html are present."""
    leaf_bytes = make_leaf_bytes()
    data = {"file": (io.BytesIO(leaf_bytes), "sample.jpg", "image/jpeg")}
    response = flask_client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()

    expected_keys = [
        "success", "plant", "status", "raw_label", "confidence",
        "confidence_score", "margin", "entropy", "is_inconclusive",
        "is_leaf", "message"
    ]
    for key in expected_keys:
        assert key in res, f"Response missing expected field: {key}"
