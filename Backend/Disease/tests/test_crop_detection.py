import io
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from main import app
from services.crop_service import validate_leaf_image, detect_crop_type

client = TestClient(app)


def create_synthetic_leaf(color=(34, 139, 34)):
    """Creates a synthetic green leaf image for testing."""
    img = Image.new("RGB", (128, 128), color=color)
    draw = ImageDraw.Draw(img)
    # Draw leaf veins
    draw.line([(64, 10), (64, 118)], fill=(20, 100, 20), width=3)
    draw.line([(64, 40), (20, 25)], fill=(20, 100, 20), width=2)
    draw.line([(64, 40), (108, 25)], fill=(20, 100, 20), width=2)
    draw.line([(64, 70), (15, 60)], fill=(20, 100, 20), width=2)
    draw.line([(64, 70), (113, 60)], fill=(20, 100, 20), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue(), img


def create_synthetic_diya_flame():
    """Creates a synthetic fire/lamp flame image with intense red/yellow and no vegetation."""
    img = Image.new("RGB", (128, 128), color=(255, 120, 0))
    draw = ImageDraw.Draw(img)
    # Bright flame center
    draw.ellipse([40, 20, 88, 90], fill=(255, 240, 50))
    draw.ellipse([50, 35, 78, 75], fill=(255, 255, 200))
    # Clay diya base
    draw.rectangle([20, 85, 108, 120], fill=(160, 60, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue(), img


def test_validate_leaf_image_on_flame():
    """Verify that a diya / flame image is identified as a non-leaf object."""
    _, flame_img = create_synthetic_diya_flame()
    is_leaf, msg, meta = validate_leaf_image(flame_img)
    assert is_leaf is False
    assert "does not appear to contain a plant leaf" in msg.lower() or "flame" in msg.lower() or "non-crop" in msg.lower()


def test_validate_leaf_image_on_real_leaf():
    """Verify that a green plant leaf is accepted."""
    _, leaf_img = create_synthetic_leaf()
    is_leaf, msg, meta = validate_leaf_image(leaf_img)
    assert is_leaf is True
    assert "verified" in msg.lower() or "valid" in msg.lower() or "confirmed" in msg.lower()


def test_detect_crop_type_on_leaf():
    """Verify crop detection returns structured candidate scores."""
    _, leaf_img = create_synthetic_leaf()
    result = detect_crop_type(leaf_img)
    assert result["is_leaf"] is True
    assert "detected_crop" in result
    assert result["confidence"] > 0


def test_api_detect_crop_endpoint():
    """Verify POST /api/v1/disease/detect-crop endpoint."""
    leaf_bytes, _ = create_synthetic_leaf()
    response = client.post(
        "/api/v1/disease/detect-crop",
        files={"image": ("test_leaf.jpg", leaf_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.get_json() if hasattr(response, "get_json") else response.json()
    assert data["is_leaf"] is True
    assert "detected_crop" in data


def test_api_analyze_rejects_diya_flame():
    """Verify POST /api/v1/disease/analyze rejects flame/diya with 400 Bad Request."""
    flame_bytes, _ = create_synthetic_diya_flame()
    response = client.post(
        "/api/v1/disease/analyze",
        data={"crop": "chilli"},
        files={"image": ("diya.jpg", flame_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.get_json() if hasattr(response, "get_json") else response.json()
    assert "detail" in data
    assert "crop leaf" in data["detail"].lower() or "detected" in data["detail"].lower()


def test_api_analyze_auto_crop():
    """Verify POST /api/v1/disease/analyze works when crop='auto'."""
    leaf_bytes, _ = create_synthetic_leaf()
    response = client.post(
        "/api/v1/disease/analyze",
        data={"crop": "auto"},
        files={"image": ("leaf.jpg", leaf_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.get_json() if hasattr(response, "get_json") else response.json()
    assert data["is_leaf"] is True
    assert data["crop"] in [
        "tomato", "chilli", "brinjal", "okra", "cucumber",
        "corn", "potato", "apple", "grape", "peach", "strawberry", "cherry"
    ]
