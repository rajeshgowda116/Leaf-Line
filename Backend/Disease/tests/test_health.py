from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify that GET /health returns 200 and valid status."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data.get("status") == "healthy"
    assert data.get("service") == "disease-detection"
    assert "version" in data
    assert "environment" in data
