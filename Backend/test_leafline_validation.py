import io
import os
import sys
import json
import pytest
import numpy as np
from PIL import Image

# Ensure Backend directory is on sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from api_backend import app, CLASS_NAMES, model, IMG_SIZE, MAX_FILE_SIZE_BYTES
from leaf_validator import validate_image_file, is_plant_leaf
from treatment_service import get_treatment_info, build_marketplace_links, is_approved_https_url


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def make_leaf_image_bytes(color=(34, 139, 34), size=(200, 200)) -> bytes:
    """Helper to generate a synthetic green leaf image as JPEG bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_diya_flame_bytes(size=(200, 200)) -> bytes:
    """Helper to generate a lamp/diya flame image (bright orange/red/yellow, no vegetation)."""
    img = Image.new("RGB", size, color=(255, 69, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_human_skin_bytes(size=(200, 200)) -> bytes:
    """Helper to generate a skin/face-like image (peach/tan, zero chlorophyll)."""
    img = Image.new("RGB", size, color=(235, 185, 155))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ============================================================================
# 1. Model Configuration & Dimensions
# ============================================================================

def test_model_class_count():
    """Verify that CLASS_NAMES exactly has 38 classes matching PlantVillage dataset."""
    assert len(CLASS_NAMES) == 38
    assert "Tomato___Early_blight" in CLASS_NAMES
    assert "Tomato___healthy" in CLASS_NAMES
    assert "Potato___Late_blight" in CLASS_NAMES
    assert "Apple___Apple_scab" in CLASS_NAMES


def test_model_output_dimensions():
    """Verify model output shape matches 38 classes."""
    assert hasattr(model, "output_shape")
    assert model.output_shape[-1] == 38


def test_input_tensor_shape_and_preprocessing():
    """Verify input preprocessing produces (1, 128, 128, 3) float tensor in [0, 1]."""
    raw_leaf = make_leaf_image_bytes()
    img = Image.open(io.BytesIO(raw_leaf)).convert("RGB")
    resized = img.resize(IMG_SIZE)
    arr = np.array(resized, dtype=np.float32) / 255.0
    tensor = np.expand_dims(arr, axis=0)

    assert tensor.shape == (1, 128, 128, 3)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


# ============================================================================
# 2. File & Format Validation (leaf_validator.validate_image_file)
# ============================================================================

def test_validate_image_file_valid_jpeg():
    valid_bytes = make_leaf_image_bytes()
    is_valid, msg, img = validate_image_file(valid_bytes)
    assert is_valid is True
    assert img is not None
    assert img.mode == "RGB"


def test_validate_image_file_empty():
    is_valid, msg, img = validate_image_file(b"")
    assert is_valid is False
    assert "empty" in msg.lower()
    assert img is None


def test_validate_image_file_corrupt():
    corrupt_bytes = b"NotAnImageHeader1234567890"
    is_valid, msg, img = validate_image_file(corrupt_bytes)
    assert is_valid is False
    assert "corrupt" in msg.lower() or "unable to read" in msg.lower()
    assert img is None


def test_validate_image_file_oversized():
    # Pass a low max_size to test without allocating 10MB in memory
    valid_bytes = make_leaf_image_bytes()
    is_valid, msg, img = validate_image_file(valid_bytes, max_size_bytes=100)
    assert is_valid is False
    assert "exceeds" in msg.lower()
    assert img is None


# ============================================================================
# 3. Binary Plant-Leaf Validation (leaf_validator.is_plant_leaf)
# ============================================================================

def test_is_plant_leaf_accepts_healthy_green_leaf():
    leaf_bytes = make_leaf_image_bytes(color=(46, 139, 87))  # Sea green
    img = Image.open(io.BytesIO(leaf_bytes)).convert("RGB")
    is_leaf, conf, msg, meta = is_plant_leaf(img)
    assert is_leaf is True
    assert "verified" in msg.lower()


def test_is_plant_leaf_rejects_diya_flame():
    """Verify that a lamp/diya flame is rejected with the exact required message."""
    diya_bytes = make_diya_flame_bytes()
    img = Image.open(io.BytesIO(diya_bytes)).convert("RGB")
    is_leaf, conf, msg, meta = is_plant_leaf(img)
    assert is_leaf is False
    expected_msg = "This image does not appear to contain a plant leaf. Please upload a clear leaf photograph."
    assert msg == expected_msg


def test_is_plant_leaf_rejects_human_skin():
    """Verify that human skin tones (peach/tan) without chlorophyll are rejected."""
    skin_bytes = make_human_skin_bytes()
    img = Image.open(io.BytesIO(skin_bytes)).convert("RGB")
    is_leaf, conf, msg, meta = is_plant_leaf(img)
    assert is_leaf is False
    assert "does not appear to contain a plant leaf" in msg


# ============================================================================
# 4. /predict Endpoint Behavior
# ============================================================================

def test_predict_endpoint_with_leaf(client):
    """Test valid leaf image upload returns HTTP 200 with structured fields."""
    leaf_bytes = make_leaf_image_bytes()
    data = {
        "file": (io.BytesIO(leaf_bytes), "tomato_leaf.jpg", "image/jpeg")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()
    assert res["success"] is True
    assert res["is_leaf"] is True
    assert "plant" in res
    assert "status" in res
    assert "raw_label" in res
    assert "confidence" in res
    assert "confidence_score" in res
    assert "margin" in res
    assert "entropy" in res
    assert "is_inconclusive" in res
    assert "message" in res


def test_predict_endpoint_rejects_diya_lamp(client):
    """Test non-leaf diya flame returns HTTP 400 with the exact rejection message."""
    diya_bytes = make_diya_flame_bytes()
    data = {
        "file": (io.BytesIO(diya_bytes), "lamp.jpg", "image/jpeg")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res = response.get_json()
    assert res["success"] is False
    assert res["is_leaf"] is False
    expected_msg = "This image does not appear to contain a plant leaf. Please upload a clear leaf photograph."
    assert res["error"] == expected_msg


def test_predict_endpoint_corrupt_file(client):
    data = {
        "file": (io.BytesIO(b"garbagebytes"), "corrupt.jpg", "image/jpeg")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res = response.get_json()
    assert res["success"] is False
    assert res["is_leaf"] is False


def test_predict_endpoint_no_file(client):
    response = client.post("/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    res = response.get_json()
    assert res["success"] is False


# ============================================================================
# 5. Uncertainty Quantification & Suppression of Agrochemicals
# ============================================================================

def test_inconclusive_suppresses_chemicals_and_marketplace():
    """Verify that when is_inconclusive=True, treatment and marketplace links are strictly omitted."""
    info = get_treatment_info("Tomato___Early_blight", is_inconclusive=True)
    assert info["status_type"] == "inconclusive"
    assert info["treatment"] == []
    assert info["active_ingredients"] == []
    assert info["marketplace_links"] == []
    assert len(info["safety_precautions"]) > 0
    assert any("not apply synthetic agrochemicals" in p.lower() for p in info["safety_precautions"])


def test_healthy_plant_suppresses_chemicals_and_marketplace():
    """Verify that healthy plant never recommends chemicals or marketplace links."""
    for healthy_label in ["Tomato___healthy", "Potato___healthy", "Apple___healthy"]:
        info = get_treatment_info(healthy_label)
        assert info["status_type"] == "healthy"
        assert info["treatment"] == []
        assert info["marketplace_links"] == []
        assert info["active_ingredients"] == []


def test_inconclusive_api_query_parameter(client):
    """Verify ?inconclusive=true query parameter on /treatment-info endpoint."""
    response = client.get("/treatment-info/Tomato___Early_blight?inconclusive=true")
    assert response.status_code == 200
    info = response.get_json()
    assert info["status_type"] == "inconclusive"
    assert info["treatment"] == []
    assert info["marketplace_links"] == []


# ============================================================================
# 6. Marketplace Links URL-Encoding & Security
# ============================================================================

def test_marketplace_links_url_encoding():
    links = build_marketplace_links(["Mancozeb 75% WP", "Copper Oxychloride"])
    assert len(links) > 0
    for link in links:
        assert "%" in link["url"] or "+" in link["url"]  # Properly encoded spaces/symbols
        assert "Mancozeb 75% WP" not in link["url"]  # Raw unencoded string not directly in query
        assert is_approved_https_url(link["url"]) is True


# ============================================================================
# 7. Database Integrity & History Endpoints
# ============================================================================

def test_all_38_classes_in_treatments_database():
    """Verify that every single one of the 38 model classes is present in treatments.json with complete fields."""
    import json
    treatments_path = os.path.join(BACKEND_DIR, "data", "treatments.json")
    with open(treatments_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    assert len(db) == 38
    for cls in CLASS_NAMES:
        assert cls in db, f"Class {cls} missing from treatments.json!"
        entry = db[cls]
        assert "crop" in entry
        assert "disease" in entry
        assert "symptoms" in entry
        assert "non_chemical_management" in entry
        assert "prevention" in entry
        assert "treatment" in entry
        assert "active_ingredients" in entry
        assert "source_references" in entry
        assert "last_reviewed" in entry

        if entry.get("is_healthy", False):
            assert entry["treatment"] == [], f"Healthy class {cls} must not contain chemical treatments!"
            assert entry["active_ingredients"] == []


def test_prediction_history_endpoints(client):
    """Verify /api/history GET and DELETE endpoints."""
    # Clear history first
    del_resp = client.delete("/api/history")
    assert del_resp.status_code == 200

    # Verify history is empty
    get_resp = client.get("/api/history")
    assert get_resp.status_code == 200
    assert get_resp.get_json()["history"] == []

    # Send a leaf prediction
    leaf_bytes = make_leaf_image_bytes()
    client.post("/predict", data={"file": (io.BytesIO(leaf_bytes), "scan1.jpg", "image/jpeg")}, content_type="multipart/form-data")

    # Send a non-leaf rejection
    diya_bytes = make_diya_flame_bytes()
    client.post("/predict", data={"file": (io.BytesIO(diya_bytes), "lamp1.jpg", "image/jpeg")}, content_type="multipart/form-data")

    # Check that history now has 2 entries
    get_resp2 = client.get("/api/history")
    assert get_resp2.status_code == 200
    history = get_resp2.get_json()["history"]
    assert len(history) == 2
    assert history[0]["filename"] == "lamp1.jpg"  # Newest first
    assert history[0]["is_leaf"] is False
    assert history[1]["filename"] == "scan1.jpg"
    assert history[1]["is_leaf"] is True
