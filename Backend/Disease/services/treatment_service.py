import json
from pathlib import Path
from typing import Optional, Dict, Any

from schemas.disease import TreatmentGuidance

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "treatment_guides.json"

_TREATMENT_DB: Optional[Dict[str, Dict[str, Any]]] = None


def load_treatment_db() -> Dict[str, Dict[str, Any]]:
    """Loads treatment guides database from JSON file."""
    global _TREATMENT_DB
    if _TREATMENT_DB is not None:
        return _TREATMENT_DB

    if not DATA_FILE.exists():
        _TREATMENT_DB = {}
        return _TREATMENT_DB

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            _TREATMENT_DB = json.load(f)
    except Exception:
        _TREATMENT_DB = {}

    return _TREATMENT_DB


def normalize_crop_name(crop: str) -> str:
    """Standardize user crop strings to internal key."""
    c = crop.strip().lower()
    if c in ("tomato", "tomatoes"):
        return "tomato"
    if c in ("chilli", "chili", "pepper", "green chilli", "hot pepper"):
        return "chilli"
    if c in ("brinjal", "eggplant", "aubergine"):
        return "brinjal"
    if c in ("okra", "ladies finger", "ladies' finger", "lady finger", "bhendi"):
        return "okra"
    if c in ("cucumber", "cucumbers", "khira"):
        return "cucumber"
    return c


def get_treatment_guidance(crop: str, disease: str) -> TreatmentGuidance:
    """
    Retrieves verified treatment and prevention guidance for a given crop and disease.

    If no exact verified guidance is available, returns an unverified placeholder
    demanding expert human verification, without hallucinating chemical recommendations.
    """
    db = load_treatment_db()
    norm_crop = normalize_crop_name(crop)
    crop_guides = db.get(norm_crop, {})

    # Try exact match first
    if disease in crop_guides:
        return TreatmentGuidance(**crop_guides[disease])

    # Try case-insensitive / partial match
    disease_lower = disease.strip().lower()
    for known_name, guide_data in crop_guides.items():
        if known_name.lower() == disease_lower:
            return TreatmentGuidance(**guide_data)

    for known_name, guide_data in crop_guides.items():
        if disease_lower in known_name.lower() or known_name.lower() in disease_lower:
            return TreatmentGuidance(**guide_data)

    # Fallback: Safe, evidence-based preliminary advice requiring expert review
    return TreatmentGuidance(
        crop=norm_crop,
        disease=disease,
        overview=(
            f"Preliminary detection for {disease} on {crop}. "
            "Specific verified agricultural guidelines are pending validation by an agronomy expert."
        ),
        symptoms=["Symptoms must be confirmed by a trained agricultural extension officer or plant pathologist."],
        prevention=[
            "Isolate suspicious plant specimens to prevent potential pathogen transmission.",
            "Avoid overhead irrigation to minimize leaf moisture duration.",
            "Sterilize pruning shears and tools with 70% alcohol or diluted bleach between plants."
        ],
        non_chemical_control=[
            "Remove visibly decaying or blighted foliage and dispose away from the production field."
        ],
        verified_treatment=[
            "No chemical recommendation provided: chemical treatments should only be applied after in-person expert confirmation to avoid phytotoxicity and resistance."
        ],
        safety_precautions=[
            "Do not apply synthetic pesticides without proper diagnosis and PPE."
        ],
        source_url="TreeTrack AI Advisory Protocol (Awaiting ICAR Reference)",
        last_verified_at="",
        verification_status="unverified",
    )
