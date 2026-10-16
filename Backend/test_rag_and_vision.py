import os
import sys
import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock
from PIL import Image

# Ensure paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, os.path.join(CURRENT_DIR, "Rag"))

from Backend.Rag.metadata_extractor import (
    detect_crop,
    detect_disease_or_pest,
    detect_category,
    extract_canonical_metadata,
)
from Backend.Rag.generate import (
    build_prompt,
    LEAFLINE_AGRONOMIST_SYSTEM_RULES,
)
from Backend.Rag.contextualizer import (
    is_greeting,
    is_follow_up_question,
    rewrite_question_with_history,
)
from Backend.Rag.relevance_gate import (
    validate_candidate_chunk,
    detect_query_crop,
)
from Backend.main import app


@pytest.fixture
def flask_client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def dummy_leaf_image_bytes():
    buf = BytesIO()
    img = Image.new("RGB", (100, 100), color=(34, 139, 34))
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ----------------------------------------------------------------
# 1. Metadata Extractor Tests
# ----------------------------------------------------------------
def test_detect_crop():
    assert detect_crop("Guidelines for tomato disease", "tomato agritech.pdf") == "Tomato"
    assert detect_crop("Bacterial wilt of brinjal plants", "brinjal dies.pdf") == "Brinjal (Eggplant)"
    assert detect_crop("Chilli anthracnose and dieback", "chilli.pdf") == "Chilli (Pepper)"
    assert detect_crop("Yellow vein mosaic virus in okra", "okra.pdf") == "Okra (Lady's Finger / Bhendi)"
    assert detect_crop("Rust and pod borer in beans", "beans 2.pdf") == "Beans"
    assert detect_crop("Powdery mildew on cucurbit vines", "cucurbits.pdf") == "Cucurbits (Cucumber / Gourds)"
    assert detect_crop("Downy mildew of spinach", "spinach.pdf") == "Spinach (Palak)"


def test_detect_disease_or_pest():
    assert "Early Blight" in detect_disease_or_pest("Leaves show dark concentric rings from early blight.")
    assert "Bacterial Wilt" in detect_disease_or_pest("Sudden wilting of foliage due to bacterial wilt.")
    assert "Powdery Mildew" in detect_disease_or_pest("White powdery coating caused by powdery mildew.")
    assert "Anthracnose" in detect_disease_or_pest("Circular sunken lesions typical of anthracnose.")


def test_detect_category():
    assert detect_category("Pathogen spores cause brown leaf spots and chlorosis") == "Disease & Pathology"
    assert detect_category("Fruit and shoot borer larvae bore into the fruit") == "Pest & Insect Management"
    assert detect_category("Sowing season, seed rate, and spacing 60x45 cm") == "Cultivation & Agronomy"


def test_extract_canonical_metadata():
    meta = extract_canonical_metadata(
        file_path="data 2/tomato agritech.pdf",
        text_content="Early blight causes concentric target-like rings on tomato leaves.",
        page=4
    )
    assert meta["crop"] == "Tomato"
    assert "Disease" in meta["category"]
    assert "Early Blight" in meta["disease_or_pest"]
    assert meta["page"] == 4


# ----------------------------------------------------------------
# 2. Prompt Builder & Grounding Rules Tests
# ----------------------------------------------------------------
def test_build_prompt_treatment_mode():
    prompt = build_prompt(
        question="How to treat and cure powdery mildew on cucurbits?",
        context="Apply neem oil spray at 3ml per liter or wettable sulfur.",
        crop="Cucurbits"
    )
    assert "LeafLine Senior Agronomist" in prompt
    assert "Organic & Bio-Control Remedies" in prompt
    assert "Reviewed Chemical Options" in prompt
    assert "Cucurbits" in prompt


def test_build_prompt_diagnosis_mode():
    prompt = build_prompt(
        question="What are the visual symptoms and signs of bacterial wilt?",
        context="Vascular browning and sudden wilting of the plant.",
        crop="Brinjal"
    )
    assert "Plant Pathologist" in prompt
    assert "Visual Symptoms Observed" in prompt
    assert "Brinjal" in prompt


def test_build_prompt_prevention_mode():
    prompt = build_prompt(
        question="How to prevent fruit borer in okra using cultural practices?",
        context="Clean weeding and crop rotation with non-hosts.",
        crop="Okra"
    )
    assert "PREVENTIVE CARE GUIDE" in prompt.upper()
    assert "Field Preparation & Cultural Practices" in prompt


# ----------------------------------------------------------------
# 3. Contextualizer & Follow-up Tests
# ----------------------------------------------------------------
def test_is_greeting():
    assert is_greeting("hello") is True
    assert is_greeting("Namaste!") is True
    assert is_greeting("good morning") is True
    assert is_greeting("How to cure early blight in tomato?") is False


def test_is_follow_up_question():
    history = [{"question": "What is early blight?", "answer": "Fungal infection."}]
    assert is_follow_up_question("How much should I spray?", history) is True
    assert is_follow_up_question("What is the organic cure for it?", history) is True
    assert is_follow_up_question("Hello", history) is False


# ----------------------------------------------------------------
# 4. Relevance Gate Tests
# ----------------------------------------------------------------
def test_relevance_gate_crop_matching():
    class MockDoc:
        def __init__(self, content, crop):
            self.page_content = content
            self.metadata = {"crop": crop}

    tomato_doc = MockDoc("Tomato early blight management", "Tomato")
    beans_doc = MockDoc("Bean anthracnose and rust", "Beans")

    # When query specifically requests Tomato
    accepted_tomato, _ = validate_candidate_chunk(tomato_doc, {"crop": "Tomato", "topic": "early blight"})
    assert accepted_tomato is True

    accepted_beans, reason = validate_candidate_chunk(beans_doc, {"crop": "Tomato", "topic": "early blight"})
    assert accepted_beans is False
    assert "Crop mismatch" in reason


# ----------------------------------------------------------------
# 5. Flask Endpoints Integration Tests
# ----------------------------------------------------------------
def test_health_endpoint(flask_client):
    resp = flask_client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "LeafLine Agritech AI"
    assert "gemini_configured" in data


def test_frontend_routes_served(flask_client):
    resp_home = flask_client.get("/")
    assert resp_home.status_code == 200

    resp_chat = flask_client.get("/chat")
    assert resp_chat.status_code == 200


def test_rag_ask_greeting(flask_client):
    resp = flask_client.post("/api/v1/rag/ask", json={"question": "Hello"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "Namaste" in data.get("answer", "") or "LeafLine" in data.get("answer", "")


def test_disease_analyze_non_plant_rejection(flask_client, dummy_leaf_image_bytes):
    mock_gemini_res = {
        "success": True,
        "configured": True,
        "model": "gemini-2.5-flash",
        "analysis": {
            "is_plant_or_leaf": False,
            "crop": "None",
            "health_status": "Non-plant",
            "diagnosis": "Not a plant or leaf image",
            "summary": "This photograph appears to be an indoor room and does not contain a crop leaf."
        }
    }

    with patch("Backend.main.analyze_plant_image_with_gemini", return_value=mock_gemini_res):
        data = {
            "image": (BytesIO(dummy_leaf_image_bytes), "room.jpg"),
            "crop": "tomato"
        }
        resp = flask_client.post("/api/v1/disease/analyze", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        result = resp.get_json()
        assert result["is_plant_or_leaf"] is False
        assert "indoor room" in result["detail"]


def test_disease_analyze_leaf_success(flask_client, dummy_leaf_image_bytes):
    mock_gemini_res = {
        "success": True,
        "configured": True,
        "model": "gemini-2.5-flash",
        "analysis": {
            "is_plant_or_leaf": True,
            "crop": "Tomato",
            "health_status": "Diseased",
            "diagnosis": "Early Blight (Alternaria solani)",
            "confidence_level": "High",
            "confidence_score": 0.96,
            "visual_symptoms": ["Concentric target-like brown spots with yellow halos"],
            "causes": "Alternaria solani fungal infection promoted by moisture.",
            "organic_treatment": ["Spray neem oil 3ml/L", "Prune lower infected leaves"],
            "chemical_treatment": ["Chlorothalonil or Mancozeb spray"],
            "preventive_measures": ["Drip irrigate at soil level", "Ensure adequate spacing"],
            "summary": "Tomato foliage shows classic symptoms of Early Blight."
        }
    }

    with patch("Backend.main.analyze_plant_image_with_gemini", return_value=mock_gemini_res):
        data = {
            "image": (BytesIO(dummy_leaf_image_bytes), "tomato.jpg"),
            "crop": "tomato"
        }
        resp = flask_client.post("/api/v1/disease/analyze", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["success"] is True
        assert result["crop_name"] == "Tomato"
        assert result["predicted_disease"] == "Early Blight (Alternaria solani)"
        assert result["status"] == "confirmed"
        assert len(result["treatment_guidance"]["symptoms"]) == 1
        assert len(result["treatment_guidance"]["non_chemical_control"]) == 2
        assert len(result["treatment_guidance"]["verified_treatment"]) == 1
