import json
import os
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Approved HTTPS marketplace hosts for security and link validation
APPROVED_MARKETPLACE_DOMAINS = {
    "amazon.in",
    "www.amazon.in",
    "flipkart.com",
    "www.flipkart.com",
    "agribegri.com",
    "www.agribegri.com",
    "bighaat.com",
    "www.bighaat.com",
}

# The 38 classes recognized by the LeafLine model in standard alphabetical order
SUPPORTED_MODEL_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

DATA_FILE = Path(__file__).resolve().parent / "data" / "treatments.json"
if not DATA_FILE.exists():
    DATA_FILE = Path(__file__).resolve().parent / "data" / "disease_treatments.json"

_DB_CACHE: Optional[Dict[str, Any]] = None


def load_knowledge_base() -> Dict[str, Any]:
    """Loads and caches the curated disease treatment knowledge base."""
    global _DB_CACHE
    if _DB_CACHE is not None:
        return _DB_CACHE

    if not DATA_FILE.exists():
        _DB_CACHE = {}
        return _DB_CACHE

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            _DB_CACHE = json.load(f)
    except Exception:
        _DB_CACHE = {}

    return _DB_CACHE


def normalize_label(raw_label: str) -> str:
    """Normalizes label string while preserving exact disease separation."""
    if not raw_label:
        return ""
    return raw_label.strip()


def is_approved_https_url(url: str) -> bool:
    """Validates that a URL is well-formed HTTPS and matches approved domains."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() != "https":
            return False
        hostname = (parsed.hostname or "").lower()
        return hostname in APPROVED_MARKETPLACE_DOMAINS
    except Exception:
        return False


def build_marketplace_links(
    active_ingredients: List[str],
    crop: str = "",
    disease: str = "",
) -> List[Dict[str, str]]:
    """
    Constructs properly URL-encoded outbound search URLs for:
    - Amazon India
    - Flipkart
    - AgriBegri (Specialized Indian Agri Portal)
    - BigHaat (Specialized Indian Agri Portal)
    """
    if not active_ingredients:
        return []

    # Use the primary active ingredient / product category for search query
    primary_query = active_ingredients[0]

    encoded_query = urllib.parse.quote_plus(primary_query)

    candidates = [
        {
            "marketplace": "Amazon India",
            "type": "General Marketplace",
            "query": primary_query,
            "url": f"https://www.amazon.in/s?k={encoded_query}",
        },
        {
            "marketplace": "Flipkart",
            "type": "General Marketplace",
            "query": primary_query,
            "url": f"https://www.flipkart.com/search?q={encoded_query}",
        },
        {
            "marketplace": "BigHaat",
            "type": "Agricultural Retailer (India)",
            "query": primary_query,
            "url": f"https://www.bighaat.com/search?q={encoded_query}",
        },
        {
            "marketplace": "AgriBegri",
            "type": "Agricultural Retailer (India)",
            "query": primary_query,
            "url": f"https://agribegri.com/search.php?search={encoded_query}",
        },
    ]

    # Validate HTTPS and approved domain security
    verified_links = [
        link for link in candidates
        if is_approved_https_url(link["url"])
    ]

    return verified_links


def get_treatment_info(raw_label: str, is_inconclusive: bool = False) -> Dict[str, Any]:
    """
    Retrieves reviewed treatment information, non-chemical management,
    and external marketplace search links for a given raw model label.

    Classifies result into:
    - 'healthy': Healthy plant, strictly NO chemical recommendations.
    - 'inconclusive': Unreliable or ambiguous prediction, strictly NO chemical recommendations.
    - 'supported_disease': Verified disease with reviewed cultural and chemical guidance.
    - 'unsupported': Known model class but detailed treatment not yet in knowledge base.
    - 'unrecognized': Label not found in model classes.
    """
    norm_label = normalize_label(raw_label)
    if not norm_label:
        return {
            "error": "Missing or empty raw_label parameter.",
            "status_type": "invalid",
        }

    # Dedicated handler for explicit inconclusive label
    if norm_label.lower() in ("inconclusive", "uncertain", "inconclusive_diagnosis"):
        return {
            "raw_label": norm_label,
            "crop": "Undetermined Crop / Plant",
            "disease": "Inconclusive Result",
            "status_type": "inconclusive",
            "overview": (
                "The analysis returned an inconclusive result because diagnostic confidence is below safety thresholds "
                "or the leaf features appear ambiguous. No agrochemicals should be applied without professional verification."
            ),
            "symptoms": [
                "Visual evidence was insufficient or mixed to make a definitive diagnosis."
            ],
            "non_chemical_management": [
                "Take another clear, well-lit photograph focusing on a single leaf blade in diffused daylight.",
                "Inspect both top and underside of the foliage for active pests or signs of fungal spores."
            ],
            "prevention": [
                "Maintain good field sanitation and crop monitoring.",
                "Ensure balanced watering and nutrient management."
            ],
            "treatment": [],  # Strictly NO chemical treatments for inconclusive
            "active_ingredients": [],
            "marketplace_links": [],  # Strictly NO product purchase links
            "safety_precautions": [
                "Do NOT apply synthetic agrochemicals based on an inconclusive diagnosis.",
                "Consult a local agricultural extension officer (KVK) for in-person crop inspection."
            ],
            "source_references": [
                "ICAR / Agricultural Extension Guidelines"
            ],
            "last_reviewed": "2026-03-15",
            "disclaimer": (
                "Prediction is inconclusive. Chemical interventions must NOT be applied without professional in-person confirmation."
            ),
        }

    kb = load_knowledge_base()

    # 1. Check if exact label is in curated knowledge base
    if norm_label in kb:
        entry = kb[norm_label]
        is_healthy = entry.get("is_healthy", False)

        if is_healthy:
            return {
                "raw_label": norm_label,
                "crop": entry.get("crop", "Plant"),
                "disease": entry.get("disease", "Healthy"),
                "status_type": "healthy",
                "overview": entry.get("overview", "Plant appears healthy with no visible disease symptoms."),
                "symptoms": entry.get("symptoms", []),
                "non_chemical_management": entry.get("non_chemical_management", []),
                "prevention": entry.get("prevention", []),
                "treatment": [],  # Never recommend chemicals for healthy plants
                "active_ingredients": [],
                "marketplace_links": [],  # No product links for healthy plants
                "safety_precautions": entry.get("safety_precautions", []),
                "source_references": entry.get("source_references", []),
                "last_reviewed": entry.get("last_reviewed", "2026-03-15"),
                "disclaimer": (
                    "Plant classified as healthy. No chemical intervention is required. "
                    "Maintain regular scouting and good agricultural practices."
                ),
            }

        # Diseased plant with reviewed treatment data
        active_ingredients = entry.get("active_ingredients", [])
        crop = entry.get("crop", "")
        disease = entry.get("disease", "")

        # If inconclusive prediction: suppress chemical recommendations and marketplace links
        if is_inconclusive:
            return {
                "raw_label": norm_label,
                "crop": crop,
                "disease": f"{disease} (Inconclusive)",
                "status_type": "inconclusive",
                "overview": (
                    f"Visual patterns show some resemblance to {crop} {disease}. However, because the prediction "
                    "confidence is below diagnostic thresholds or exhibits high entropy, this result is marked as inconclusive."
                ),
                "symptoms": entry.get("symptoms", []),
                "non_chemical_management": entry.get("non_chemical_management", []),
                "prevention": entry.get("prevention", []),
                "treatment": [],  # Strictly NO chemical treatments for inconclusive predictions
                "active_ingredients": [],
                "marketplace_links": [],  # Strictly NO product purchase links for inconclusive predictions
                "safety_precautions": [
                    "Do NOT apply synthetic agrochemicals based on an inconclusive diagnosis.",
                    "Consult a local agricultural extension officer (KVK) for in-person crop inspection."
                ],
                "source_references": entry.get("source_references", []),
                "last_reviewed": entry.get("last_reviewed", "2026-03-15"),
                "disclaimer": (
                    "Prediction is inconclusive. Chemical interventions must NOT be applied without professional in-person confirmation."
                ),
            }

        marketplace_links = build_marketplace_links(active_ingredients, crop=crop, disease=disease)

        return {
            "raw_label": norm_label,
            "crop": crop,
            "disease": disease,
            "status_type": "supported_disease",
            "overview": entry.get("overview", ""),
            "symptoms": entry.get("symptoms", []),
            "non_chemical_management": entry.get("non_chemical_management", []),
            "prevention": entry.get("prevention", []),
            "treatment": entry.get("treatment", []),
            "active_ingredients": active_ingredients,
            "marketplace_links": marketplace_links,
            "safety_precautions": entry.get("safety_precautions", []),
            "source_references": entry.get("source_references", []),
            "last_reviewed": entry.get("last_reviewed", "2026-03-15"),
            "disclaimer": (
                "This is an agricultural information service, not a substitute for a professional in-person agronomy diagnosis. "
                "Marketplace search links are provided for convenience; listings, prices, availability, and sellers are controlled "
                "by external marketplaces. Always verify product labels, approved crops, and safety intervals before application."
            ),
        }

    # 2. Check if it is a recognized model class without detailed reviewed entry
    if norm_label in SUPPORTED_MODEL_CLASSES:
        # Parse basic plant and condition from label
        parts = norm_label.split("___", 1)
        crop_guess = parts[0].replace("_", " ").title() if len(parts) > 0 else "Plant"
        cond_guess = parts[1].replace("_", " ").title() if len(parts) > 1 else "Condition"

        is_healthy = "healthy" in cond_guess.lower()

        if is_healthy:
            return {
                "raw_label": norm_label,
                "crop": crop_guess,
                "disease": "Healthy",
                "status_type": "healthy",
                "overview": (
                    f"The {crop_guess} foliage appears healthy with no visible pathogen symptoms detected."
                ),
                "symptoms": [
                    "Normal green coloration and turgid leaves.",
                    "No visible lesions, spots, or chlorosis."
                ],
                "non_chemical_management": [
                    "Maintain proper irrigation and avoid waterlogging or drought stress.",
                    "Ensure adequate canopy spacing for ventilation.",
                    "Conduct regular weekly scouting for early pest or disease detection."
                ],
                "prevention": [
                    "Practice standard crop rotation and weed management.",
                    "Sanitize pruning equipment before and after field use.",
                    "Apply balanced organic compost to maintain root health."
                ],
                "treatment": [],  # Strictly NO chemical treatments for healthy plants
                "active_ingredients": [],
                "marketplace_links": [],  # Strictly NO product links for healthy plants
                "safety_precautions": [
                    "No chemical interventions are necessary or recommended for healthy crops."
                ],
                "source_references": [
                    "ICAR / Agricultural Extension Guidelines"
                ],
                "last_reviewed": "2026-03-15",
                "disclaimer": (
                    "Plant classified as healthy. No chemical intervention is required. "
                    "Maintain regular scouting and good agricultural practices."
                ),
            }

        return {
            "raw_label": norm_label,
            "crop": crop_guess,
            "disease": cond_guess,
            "status_type": "unsupported",
            "overview": (
                f"Prediction indicates {crop_guess} - {cond_guess}. Detailed treatment guidance and product mappings "
                f"for this specific class are currently under review."
            ),
            "symptoms": [
                f"Visual indicators associated with {cond_guess} on {crop_guess}."
            ],
            "non_chemical_management": [
                "Isolate suspicious plant specimens to prevent potential transmission.",
                "Ensure clean tool sanitation and avoid working among plants when wet."
            ],
            "prevention": [
                "Practice good agricultural sanitation and crop rotation.",
                "Monitor crops closely for symptom progression."
            ],
            "treatment": [],  # Do NOT invent unverified chemicals
            "active_ingredients": [],
            "marketplace_links": [],  # No unverified links
            "safety_precautions": [
                "Please consult a local agricultural extension officer or certified agronomist for in-person confirmation."
            ],
            "source_references": [
                "ICAR / Agricultural Extension Guidelines"
            ],
            "last_reviewed": "",
            "disclaimer": (
                "Specific chemical and marketplace recommendations for this class have not yet been verified. "
                "Do not apply synthetic agrochemicals without professional confirmation."
            ),
        }

    # 3. Completely unrecognized label
    return {
        "raw_label": norm_label,
        "crop": "Unknown",
        "disease": "Unrecognized Condition",
        "status_type": "unrecognized",
        "overview": (
            f"The label '{norm_label}' is not a recognized class of the LeafLine model. "
            "Unable to retrieve treatment or management information."
        ),
        "symptoms": [],
        "non_chemical_management": [],
        "prevention": [],
        "treatment": [],
        "active_ingredients": [],
        "marketplace_links": [],
        "safety_precautions": [],
        "source_references": [],
        "last_reviewed": "",
        "disclaimer": "Unrecognized label. Please re-run leaf analysis with a clear photograph.",
    }
