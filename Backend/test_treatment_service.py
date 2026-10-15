import os
import sys
import pytest
from urllib.parse import urlparse

# Ensure Backend directory is on sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from treatment_service import (
    get_treatment_info,
    build_marketplace_links,
    is_approved_https_url,
    normalize_label,
    APPROVED_MARKETPLACE_DOMAINS,
    SUPPORTED_MODEL_CLASSES
)
from api_backend import app


# -------------------------------------------------------------
# 1. Tests for treatment_service logic
# -------------------------------------------------------------

def test_supported_disease_tomato_early_blight():
    """Verify supported disease returns complete reviewed guidance and marketplace links."""
    result = get_treatment_info("Tomato___Early_blight")
    assert result["status_type"] == "supported_disease"
    assert result["crop"] == "Tomato"
    assert "Early Blight" in result["disease"]
    assert len(result["symptoms"]) > 0
    assert len(result["non_chemical_management"]) > 0
    assert len(result["prevention"]) > 0
    assert len(result["treatment"]) > 0
    assert len(result["active_ingredients"]) > 0
    assert len(result["marketplace_links"]) > 0
    assert len(result["source_references"]) > 0
    assert result["last_reviewed"] != ""
    assert "disclaimer" in result

    # Check marketplace link integrity
    marketplaces = {link["marketplace"] for link in result["marketplace_links"]}
    assert "Amazon India" in marketplaces
    assert "Flipkart" in marketplaces
    assert "BigHaat" in marketplaces
    assert "AgriBegri" in marketplaces

    for link in result["marketplace_links"]:
        parsed = urlparse(link["url"])
        assert parsed.scheme == "https"
        assert parsed.hostname.lower() in APPROVED_MARKETPLACE_DOMAINS


def test_healthy_plant_tomato_no_chemicals():
    """Verify healthy tomato returns no chemicals and no product purchase links."""
    result = get_treatment_info("Tomato___healthy")
    assert result["status_type"] == "healthy"
    assert result["crop"] == "Tomato"
    assert "healthy" in result["disease"].lower()
    assert result["treatment"] == []
    assert result["active_ingredients"] == []
    assert result["marketplace_links"] == []
    assert len(result["non_chemical_management"]) > 0
    assert "disclaimer" in result


def test_healthy_plant_arbitrary_model_class_no_chemicals():
    """Verify any model healthy class (e.g. Blueberry) returns healthy status with 0 chemicals."""
    result = get_treatment_info("Blueberry___healthy")
    assert result["status_type"] == "healthy"
    assert result["treatment"] == []
    assert result["marketplace_links"] == []
    assert result["active_ingredients"] == []


def test_unsupported_model_class(monkeypatch):
    """Verify recognized model class without detailed KB entry explains limitation safely."""
    import treatment_service
    orig_kb = treatment_service.load_knowledge_base()
    mock_kb = {k: v for k, v in orig_kb.items() if k != "Corn___Northern_Leaf_Blight"}
    monkeypatch.setattr(treatment_service, "load_knowledge_base", lambda: mock_kb)

    result = get_treatment_info("Corn___Northern_Leaf_Blight")
    assert result["status_type"] == "unsupported"
    assert result["crop"] == "Corn"
    assert result["treatment"] == []  # MUST NOT invent chemicals
    assert result["marketplace_links"] == []  # MUST NOT generate fake links
    assert "under review" in result["overview"].lower()


def test_unrecognized_label():
    """Verify unrecognized random label returns unrecognized status."""
    result = get_treatment_info("FakeCrop___Imaginary_Rot")
    assert result["status_type"] == "unrecognized"
    assert result["treatment"] == []
    assert result["marketplace_links"] == []


def test_empty_and_whitespace_label():
    """Verify empty or whitespace-only label is handled as invalid."""
    assert get_treatment_info("")["status_type"] == "invalid"
    assert get_treatment_info("   ")["status_type"] == "invalid"


def test_build_marketplace_links_url_encoding():
    """Verify proper URL encoding of active ingredient queries with special characters."""
    links = build_marketplace_links(["Mancozeb 75% WP fungicide"])
    assert len(links) > 0
    for link in links:
        url = link["url"]
        # Special character '%' and spaces should be safely encoded
        assert "%" not in link["query"] or "%25" in url or "75%25" in url or "75%2B" in url or "+" in url
        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.hostname.lower() in APPROVED_MARKETPLACE_DOMAINS


def test_is_approved_https_url_security():
    """Verify malicious or unapproved URLs are rejected."""
    assert is_approved_https_url("https://www.amazon.in/s?k=test") is True
    assert is_approved_https_url("https://www.flipkart.com/search?q=test") is True
    assert is_approved_https_url("https://bighaat.com/search?q=test") is True
    assert is_approved_https_url("http://www.amazon.in/s?k=test") is False  # Insecure HTTP
    assert is_approved_https_url("https://malicious-site.com/phish") is False  # Unapproved domain
    assert is_approved_https_url("javascript:alert(1)") is False  # Script injection
    assert is_approved_https_url("") is False


# -------------------------------------------------------------
# 2. Tests for Flask API endpoints
# -------------------------------------------------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_treatment_info_supported(client):
    """Test GET /treatment-info/<label> for supported disease."""
    response = client.get("/treatment-info/Tomato___Early_blight")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status_type"] == "supported_disease"
    assert data["crop"] == "Tomato"
    assert len(data["marketplace_links"]) > 0


def test_api_treatment_info_healthy(client):
    """Test GET /treatment-info/<label> for healthy plant."""
    response = client.get("/treatment-info/Tomato___healthy")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status_type"] == "healthy"
    assert data["treatment"] == []
    assert data["marketplace_links"] == []


def test_api_treatment_info_unsupported(client, monkeypatch):
    """Test GET /treatment-info/<label> for unsupported model class."""
    import treatment_service
    orig_kb = treatment_service.load_knowledge_base()
    mock_kb = {k: v for k, v in orig_kb.items() if k != "Corn___Northern_Leaf_Blight"}
    monkeypatch.setattr(treatment_service, "load_knowledge_base", lambda: mock_kb)

    response = client.get("/treatment-info/Corn___Northern_Leaf_Blight")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status_type"] == "unsupported"


def test_api_treatment_info_unrecognized(client):
    """Test GET /treatment-info/<label> for unrecognized disease returns 404."""
    response = client.get("/treatment-info/Completely_Unknown_Label")
    assert response.status_code == 404
    data = response.get_json()
    assert data["status_type"] == "unrecognized"


def test_api_treatment_info_empty(client):
    """Test GET /treatment-info/ returns 400 Bad Request."""
    response = client.get("/treatment-info/")
    assert response.status_code == 400


def test_api_root_serves_html(client):
    """Test GET / serves frontend index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"LeafLine" in response.data


def test_api_about_serves_html(client):
    """Test GET /about serves frontend about.html."""
    response = client.get("/about")
    assert response.status_code == 200


def test_api_predict_missing_file(client):
    """Test POST /predict without file returns 400."""
    response = client.post("/predict")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
