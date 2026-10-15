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
    Supports specialized 5-crop guides, multi-crop treatments database, and inconclusive handling.
    """
    norm_crop = normalize_crop_name(crop)
    disease_lower = disease.strip().lower()

    # 1. Inconclusive or ambiguous diagnosis: strictly NO chemicals
    if "inconclusive" in disease_lower or "uncertain" in disease_lower:
        return TreatmentGuidance(
            crop=norm_crop,
            disease="Inconclusive Diagnosis",
            overview=(
                f"Visual patterns on this {norm_crop} specimen are ambiguous or model confidence is below "
                "diagnostic thresholds. Agrochemicals must not be applied without in-person confirmation by a plant pathologist."
            ),
            symptoms=["Symptoms could not be identified with conclusive statistical confidence."],
            prevention=[
                "Maintain proper crop sanitation and monitor adjacent plants for symptoms.",
                "Ensure adequate spacing and avoid overhead watering that keeps foliage wet."
            ],
            non_chemical_control=[
                "Inspect foliage in diffused natural daylight and check for active insect vectors or fungal spores.",
                "Quarantine affected plants if symptoms appear to be spreading."
            ],
            verified_treatment=[
                "No chemical treatment recommended: applying synthetic pesticides without confirmation is hazardous to crop health."
            ],
            safety_precautions=[
                "Do NOT apply synthetic agrochemicals based on an inconclusive diagnosis.",
                "Consult a local agricultural extension officer (KVK) for in-person crop inspection."
            ],
            source_url="TreeTrack AI / ICAR Diagnostic Safety Protocol",
            last_verified_at="2026-03-15",
            verification_status="inconclusive",
        )

    # 2. Check specialized treatment guides
    db = load_treatment_db()
    crop_guides = db.get(norm_crop, {})

    if disease in crop_guides:
        return TreatmentGuidance(**crop_guides[disease])

    for known_name, guide_data in crop_guides.items():
        if known_name.lower() == disease_lower:
            return TreatmentGuidance(**guide_data)

    for known_name, guide_data in crop_guides.items():
        if disease_lower in known_name.lower() or known_name.lower() in disease_lower:
            return TreatmentGuidance(**guide_data)

    # 3. Look up from multi-crop treatments.json (covering all 38 classes)
    treatments_path = Path(__file__).resolve().parent.parent.parent / "data" / "treatments.json"
    if treatments_path.exists():
        try:
            with open(treatments_path, "r", encoding="utf-8") as f:
                multi_db = json.load(f)

            # Find matching entry by crop and disease
            for raw_label, entry in multi_db.items():
                e_crop = entry.get("crop", "").lower()
                e_disease = entry.get("disease", "").lower()

                if e_crop == norm_crop and (e_disease == disease_lower or disease_lower in e_disease or e_disease in disease_lower):
                    is_h = entry.get("is_healthy", False) or "healthy" in e_disease
                    return TreatmentGuidance(
                        crop=entry.get("crop", norm_crop.title()),
                        disease=entry.get("disease", disease),
                        overview=entry.get("overview", f"Pathology information for {entry.get('disease')} on {entry.get('crop')}."),
                        symptoms=entry.get("symptoms", []),
                        prevention=entry.get("prevention", []),
                        non_chemical_control=entry.get("non_chemical_management", []),
                        verified_treatment=[] if is_h else entry.get("treatment", []),
                        safety_precautions=entry.get("safety_precautions", []),
                        source_url=(entry.get("source_references") or ["ICAR Guidelines"])[0],
                        last_verified_at=entry.get("last_reviewed", "2026-03-15"),
                        verification_status="verified" if not is_h else "healthy",
                    )
        except Exception:
            pass

    # 4. Fallback: Safe preliminary advice requiring expert review
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
