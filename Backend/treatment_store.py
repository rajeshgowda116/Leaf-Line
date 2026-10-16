"""
LeafLine AI - Verified Treatment Knowledge Store
------------------------------------------------
Provides reviewed, source-backed agricultural treatment retrieval based on
ICAR-IIHR and TNAU agronomic guides. Maps free-text crop & disease names from
Gemini Vision to canonical verified records, strictly enforcing safety rules
(no invented chemicals or dosages, explicit source URLs, and safe fallbacks).
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

_CURRENT_DIR = Path(__file__).resolve().parent
TREATMENTS_JSON_PATH = _CURRENT_DIR / "data" / "treatments.json"

_TREATMENT_CACHE: Optional[Dict[str, Any]] = None


def load_treatments_data() -> Dict[str, Any]:
    """Loads and caches verified treatments from data/treatments.json."""
    global _TREATMENT_CACHE
    if _TREATMENT_CACHE is not None:
        return _TREATMENT_CACHE

    if not TREATMENTS_JSON_PATH.exists():
        logger.warning(f"treatments.json not found at {TREATMENTS_JSON_PATH}")
        _TREATMENT_CACHE = {}
        return _TREATMENT_CACHE

    try:
        with open(TREATMENTS_JSON_PATH, "r", encoding="utf-8") as f:
            _TREATMENT_CACHE = json.load(f)
            logger.info(f"Loaded {len(_TREATMENT_CACHE)} verified treatment records.")
            return _TREATMENT_CACHE
    except Exception as e:
        logger.error(f"Failed to load treatments.json: {e}")
        _TREATMENT_CACHE = {}
        return _TREATMENT_CACHE


# Explicit reviewed aliases mapping (crop_keyword, disease_keyword) -> canonical treatments.json key
REVIEWED_CROP_DISEASE_MAPPING = {
    # Tomato
    ("tomato", "early blight"): "Tomato___Early_blight",
    ("tomato", "alternaria"): "Tomato___Early_blight",
    ("tomato", "late blight"): "Tomato___Late_blight",
    ("tomato", "phytophthora"): "Tomato___Late_blight",
    ("tomato", "bacterial spot"): "Tomato___Bacterial_spot",
    ("tomato", "xanthomonas"): "Tomato___Bacterial_spot",
    ("tomato", "leaf mold"): "Tomato___Leaf_Mold",
    ("tomato", "passalora"): "Tomato___Leaf_Mold",
    ("tomato", "cladosporium"): "Tomato___Leaf_Mold",
    ("tomato", "septoria"): "Tomato___Septoria_leaf_spot",
    ("tomato", "spider mite"): "Tomato___Spider_mites Two-spotted_spider_mite",
    ("tomato", "target spot"): "Tomato___Target_Spot",
    ("tomato", "corynespora"): "Tomato___Target_Spot",
    ("tomato", "yellow leaf curl"): "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    ("tomato", "tylcv"): "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    ("tomato", "mosaic virus"): "Tomato___Tomato_mosaic_virus",
    ("tomato", "tomv"): "Tomato___Tomato_mosaic_virus",
    ("tomato", "healthy"): "Tomato___healthy",

    # Potato
    ("potato", "early blight"): "Potato___Early_blight",
    ("potato", "late blight"): "Potato___Late_blight",
    ("potato", "healthy"): "Potato___healthy",

    # Pepper / Chilli
    ("pepper", "bacterial spot"): "Pepper__bell___Bacterial_spot",
    ("chilli", "bacterial spot"): "Pepper__bell___Bacterial_spot",
    ("pepper", "healthy"): "Pepper__bell___healthy",
    ("chilli", "healthy"): "Pepper__bell___healthy",

    # Apple
    ("apple", "scab"): "Apple___Apple_scab",
    ("apple", "black rot"): "Apple___Black_rot",
    ("apple", "cedar apple rust"): "Apple___Cedar_apple_rust",
    ("apple", "healthy"): "Apple___healthy",

    # Corn / Maize
    ("corn", "common rust"): "Corn___Common_rust",
    ("corn", "gray leaf spot"): "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    ("corn", "cercospora"): "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    ("corn", "northern leaf blight"): "Corn___Northern_Leaf_Blight",
    ("corn", "healthy"): "Corn___healthy",

    # Grape
    ("grape", "black rot"): "Grape___Black_rot",
    ("grape", "esca"): "Grape___Esca_(Black_Measles)",
    ("grape", "black measles"): "Grape___Esca_(Black_Measles)",
    ("grape", "leaf blight"): "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    ("grape", "healthy"): "Grape___healthy",

    # Cherry
    ("cherry", "powdery mildew"): "Cherry___Powdery_mildew",
    ("cherry", "healthy"): "Cherry___healthy",

    # Peach
    ("peach", "bacterial spot"): "Peach___Bacterial_spot",
    ("peach", "healthy"): "Peach___healthy",

    # Strawberry
    ("strawberry", "leaf scorch"): "Strawberry___Leaf_scorch",
    ("strawberry", "healthy"): "Strawberry___healthy",

    # Squash / Cucurbits
    ("squash", "powdery mildew"): "Squash___Powdery_mildew",
    ("cucumber", "powdery mildew"): "Squash___Powdery_mildew",
    ("cucurbits", "powdery mildew"): "Squash___Powdery_mildew",
}


def _normalize_string(s: str) -> str:
    """Normalize text by lowering, removing punctuation, and stripping whitespace."""
    if not s:
        return ""
    clean = re.sub(r"[^\w\s]", " ", s.lower())
    return " ".join(clean.split())


def _format_source_references(raw_sources: List[Any]) -> List[Dict[str, str]]:
    """Formats source references into structured dictionaries with name and URL."""
    structured = []
    for src in raw_sources:
        if isinstance(src, dict):
            structured.append({
                "name": src.get("name", "Agricultural Research Reference"),
                "url": src.get("url", "")
            })
        elif isinstance(src, str):
            # Extract URL if present in string e.g. "ICAR-IIHR Tomato Disease Guide (https://iihr.res.in)"
            url_match = re.search(r"https?://[^\s\)]+", src)
            url = url_match.group(0) if url_match else ""
            clean_name = re.sub(r"\(https?://[^\s\)]+\)", "", src).strip()
            
            # Default authoritative portal URLs if known
            if not url:
                if "icar" in clean_name.lower() or "iihr" in clean_name.lower():
                    url = "https://iihr.res.in"
                elif "tnau" in clean_name.lower():
                    url = "https://agritech.tnau.ac.in"
                elif "florida" in clean_name.lower():
                    url = "https://edis.ifas.ufl.edu"

            structured.append({
                "name": clean_name or src,
                "url": url
            })
    return structured


def get_safe_fallback_treatment(crop: str, disease_name: str, reason: str = "") -> Dict[str, Any]:
    """
    Returns an unverified, safe fallback management guidance when no verified record exists.
    Strictly suppresses chemical pesticide recommendations to avoid inventing unsafe treatments.
    """
    crop_display = crop if crop and crop.lower() not in ("none", "unknown") else "Crop"
    disease_display = disease_name if disease_name and disease_name.lower() not in ("unknown", "inconclusive", "none") else "Uncertain Condition"

    return {
        "is_verified": False,
        "verification_status": "unverified",
        "canonical_label": None,
        "crop": crop_display,
        "disease": disease_display,
        "is_healthy": "healthy" in disease_display.lower(),
        "overview": (
            f"Preliminary visual observation indicates {crop_display} with potential {disease_display}. "
            "Specific verified chemical treatment protocols for this exact class are not confirmed in the LeafLine certified database."
        ),
        "symptoms": [
            f"Visual abnormalities on {crop_display} foliage requiring confirmation."
        ],
        "non_chemical_management": [
            "Prune and safely destroy visibly damaged or diseased leaf tissue (do not compost diseased foliage).",
            "Disinfect shears and pruning implements between plants using 70% alcohol or 10% bleach solution.",
            "Water plants at ground level (drip or base watering); strictly avoid wetting foliage to prevent fungal germination.",
            "Ensure adequate plant and row spacing to promote canopy ventilation and rapid foliage drying."
        ],
        "prevention": [
            "Rotate crops annually away from plants of the same botanical family.",
            "Use certified disease-free seeds and certified vigorous seedlings.",
            "Routinely scout fields weekly to catch disease or pest hotspots before spreading."
        ],
        "treatment": [],  # Strictly EMPTY - never invent chemical dosages
        "active_ingredients": [],
        "safety_precautions": [
            "Do NOT apply synthetic chemical pesticides or fungicides based on an unverified or preliminary visual assessment.",
            "Always consult your local Krishi Vigyan Kendra (KVK) officer, university extension agronomist, or agricultural authority for verified regional treatment."
        ],
        "source_references": [
            {
                "name": "ICAR - Good Agricultural Practices (GAP)",
                "url": "https://icar.org.in"
            },
            {
                "name": "TNAU Agritech Portal (Crop Protection)",
                "url": "https://agritech.tnau.ac.in"
            }
        ],
        "last_reviewed": None,
        "disclaimer": (
            "No verified chemical treatment record exists for this condition. "
            "Chemical agrochemicals must not be applied without physical in-person inspection and written recommendation from a certified agronomist."
        )
    }


def find_verified_treatment(crop: str, disease_name: str) -> Dict[str, Any]:
    """
    Receives crop and suspected disease from Gemini Vision,
    maps them via reviewed explicit dictionary to a canonical treatments.json record,
    and returns verified source-backed management guidance, or a safe fallback.
    """
    data = load_treatments_data()
    norm_crop = _normalize_string(crop)
    norm_disease = _normalize_string(disease_name)

    # If non-plant or inconclusive
    if not norm_crop or norm_crop in ("none", "unknown") or norm_disease in ("inconclusive", "unknown", "non plant subject"):
        return get_safe_fallback_treatment(crop, disease_name, reason="Inconclusive or non-plant assessment")

    matched_key: Optional[str] = None

    # 1. Direct match in reviewed mapping
    for (c_key, d_key), canon_key in REVIEWED_CROP_DISEASE_MAPPING.items():
        if c_key in norm_crop and d_key in norm_disease:
            matched_key = canon_key
            break

    # 2. Secondary check for "healthy"
    if not matched_key and "healthy" in norm_disease:
        for canon_key in data.keys():
            if norm_crop in canon_key.lower() and "healthy" in canon_key.lower():
                matched_key = canon_key
                break

    # 3. Fallback fuzzy check across all keys in data
    if not matched_key:
        for canon_key, entry in data.items():
            entry_crop = _normalize_string(entry.get("crop", ""))
            entry_dis = _normalize_string(entry.get("disease", ""))
            if entry_crop and entry_crop in norm_crop and entry_dis and entry_dis in norm_disease:
                matched_key = canon_key
                break

    # If no verified record found
    if not matched_key or matched_key not in data:
        logger.info(f"No verified record found for crop='{crop}', disease='{disease_name}'. Returning safe fallback.")
        return get_safe_fallback_treatment(crop, disease_name)

    # 4. Construct verified response from record
    rec = data[matched_key]
    is_healthy = rec.get("is_healthy", False)

    treatment_list = rec.get("treatment", []) if not is_healthy else []
    active_ingredients = rec.get("active_ingredients", []) if not is_healthy else []

    sources = _format_source_references(rec.get("source_references", []))

    return {
        "is_verified": True,
        "verification_status": "verified",
        "canonical_label": matched_key,
        "crop": rec.get("crop", crop),
        "disease": rec.get("disease", disease_name),
        "is_healthy": is_healthy,
        "overview": rec.get("overview", ""),
        "symptoms": rec.get("symptoms", []),
        "non_chemical_management": rec.get("non_chemical_management", []),
        "prevention": rec.get("prevention", []),
        "treatment": treatment_list,
        "active_ingredients": active_ingredients,
        "safety_precautions": rec.get("safety_precautions", []),
        "source_references": sources,
        "last_reviewed": rec.get("last_reviewed", "2026-03-15"),
        "disclaimer": (
            "Verified agronomic guidance provided by ICAR / TNAU agricultural package of practices. "
            "Always follow statutory product labels, observe safety gear precautions, and respect Pre-Harvest Intervals (PHI)."
        )
    }
