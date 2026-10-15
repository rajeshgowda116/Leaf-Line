import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from services.image_service import validate_and_open_image, preprocess_for_model
from fastapi import HTTPException

client = TestClient(app)


def create_test_image_bytes(format="JPEG", size=(100, 100), color="green") -> bytes:
    """Helper to generate in-memory dummy test images."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


def test_validate_and_open_image_valid():
    """Verify valid JPEG is decoded to RGB PIL Image."""
    img_bytes = create_test_image_bytes(format="JPEG")
    img = validate_and_open_image(img_bytes)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_validate_and_open_image_png():
    """Verify valid PNG is decoded to RGB PIL Image."""
    img_bytes = create_test_image_bytes(format="PNG")
    img = validate_and_open_image(img_bytes)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_validate_and_open_image_empty():
    """Verify empty bytes raise HTTP 400."""
    with pytest.raises(HTTPException) as exc_info:
        validate_and_open_image(b"")
    assert exc_info.value.status_code == 400


def test_validate_and_open_image_oversized():
    """Verify oversized image raises HTTP 413."""
    dummy_data = b"0" * 1500
    with pytest.raises(HTTPException) as exc_info:
        validate_and_open_image(dummy_data, max_size_bytes=1000)
    assert exc_info.value.status_code == 413


def test_validate_and_open_image_corrupt():
    """Verify corrupt bytes raise HTTP 400."""
    with pytest.raises(HTTPException) as exc_info:
        validate_and_open_image(b"not-an-image-data-here")
    assert exc_info.value.status_code == 400


def test_preprocess_tensor_shape():
    """Verify preprocessed tensor has shape (1, 3, 224, 224)."""
    img = Image.new("RGB", (300, 300), color="green")
    tensor = preprocess_for_model(img, target_size=(224, 224))
    assert tensor.shape == (1, 3, 224, 224)
