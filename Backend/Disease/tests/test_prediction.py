import io
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def create_test_image(format="JPEG", color="green") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=color)
    img.save(buf, format=format)
    return buf.getvalue()


def test_list_supported_crops():
    """Verify GET /api/v1/disease/crops returns all 5 required crops."""
    response = client.get("/api/v1/disease/crops")
    assert response.status_code == 200
    data = response.json()
    assert "crops" in data
    crop_names = [c["name"] for c in data["crops"]]
    for required in ["tomato", "chilli", "brinjal", "okra", "cucumber"]:
        assert required in crop_names


def test_get_crop_classes_tomato():
    """Verify GET /api/v1/disease/crops/tomato/classes returns 13 classes."""
    response = client.get("/api/v1/disease/crops/tomato/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "tomato"
    assert data["num_classes"] == 13
    assert "Early Blight" in data["classes"]
    assert "Healthy" in data["classes"]


def test_get_crop_classes_unsupported():
    """Verify 404 for unsupported crop."""
    response = client.get("/api/v1/disease/crops/papaya/classes")
    assert response.status_code == 404


def test_analyze_valid_leaf():
    """Verify POST /api/v1/disease/analyze returns complete diagnosis schema."""
    img_bytes = create_test_image(format="JPEG", color="green")
    response = client.post(
        "/api/v1/disease/analyze",
        data={"crop": "tomato"},
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "tomato"
    assert "predicted_disease" in data
    assert "confidence" in data
    assert "treatment_guidance" in data
    assert "status" in data
    assert data["human_review_required"] is True
    assert len(data["source_references"]) > 0


def test_analyze_unsupported_crop():
    """Verify 400 for unsupported crop in analyze endpoint."""
    img_bytes = create_test_image(format="JPEG")
    response = client.post(
        "/api/v1/disease/analyze",
        data={"crop": "mango"},
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 400


def test_analyze_invalid_image_content():
    """Verify rejection when file content is not a real image."""
    fake_bytes = b"This is not a real leaf photograph"
    response = client.post(
        "/api/v1/disease/analyze",
        data={"crop": "tomato"},
        files={"image": ("fake.jpg", fake_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
