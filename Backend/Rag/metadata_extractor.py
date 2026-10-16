import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

# Canonical Crop Mapping based on filenames and content
CROP_NAME_MAP = {
    "tomato": "Tomato",
    "brinjal": "Brinjal (Eggplant)",
    "eggplant": "Brinjal (Eggplant)",
    "chilli": "Chilli (Pepper)",
    "chillies": "Chilli (Pepper)",
    "pepper": "Chilli (Pepper)",
    "okra": "Okra (Lady's Finger / Bhendi)",
    "bhendi": "Okra (Lady's Finger / Bhendi)",
    "bean": "Beans",
    "beans": "Beans",
    "cucurbit": "Cucurbits (Cucumber / Gourds)",
    "cucurbits": "Cucurbits (Cucumber / Gourds)",
    "cucumber": "Cucurbits (Cucumber / Gourds)",
    "gourd": "Cucurbits (Cucumber / Gourds)",
    "spinach": "Spinach (Palak)",
    "palak": "Spinach (Palak)",
}

# Major Plant Diseases and Pathogens
DISEASE_KEYWORDS = {
    "early blight": "Early Blight (Alternaria)",
    "late blight": "Late Blight (Phytophthora)",
    "bacterial wilt": "Bacterial Wilt (Ralstonia)",
    "bacterial spot": "Bacterial Spot (Xanthomonas)",
    "powdery mildew": "Powdery Mildew",
    "downy mildew": "Downy Mildew",
    "damping off": "Damping Off (Pythium / Rhizoctonia)",
    "anthracnose": "Anthracnose (Colletotrichum)",
    "leaf curl": "Leaf Curl Virus",
    "mosaic virus": "Mosaic Virus",
    "yellow vein mosaic": "Yellow Vein Mosaic Virus (YVMV)",
    "little leaf": "Little Leaf of Brinjal",
    "cercospora": "Cercospora Leaf Spot",
    "septoria": "Septoria Leaf Spot",
    "collar rot": "Collar Rot / Root Rot",
    "fruit borer": "Fruit and Shoot Borer (Leucinodes / Helicoverpa)",
    "whitefly": "Whitefly (Bemisia tabaci)",
    "aphid": "Aphids",
    "mite": "Red Spider Mite / Broad Mite",
    "thrips": "Thrips",
    "nematode": "Root Knot Nematode (Meloidogyne)",
}

# Agricultural Topic Categories
CATEGORY_PATTERNS = [
    (r"\b(disease|blight|wilt|mildew|rot|pathogen|fungus|fungal|bacteria|bacterial|virus|viral|symptom|chlorosis|necrosis)\b", "Disease & Pathology"),
    (r"\b(pest|borer|whitefly|aphid|thrips|mite|caterpillar|larva|insect|infestation)\b", "Pest & Insect Management"),
    (r"\b(treatment|control|management|fungicide|insecticide|neem|spray|bio-control|trichoderma|pseudomonas|chemical|mancozeb|copper)\b", "Plant Protection & Treatments"),
    (r"\b(cultivation|variety|varieties|sowing|nursery|seed|season|climate|soil|transplanting|spacing|irrigation|fertilizer|manure|npk)\b", "Cultivation & Agronomy"),
    (r"\b(harvest|yield|picking|post-harvest|storage|grading|packing)\b", "Harvest & Post-Harvest"),
]


def detect_crop(text: str, filename: str = "") -> str:
    """Detect primary agricultural crop from filename and content."""
    combined = f"{filename} {text}".lower()
    for key, crop_name in CROP_NAME_MAP.items():
        if re.search(rf"\b{re.escape(key)}\b", combined):
            return crop_name
    return "General Horticulture"


def detect_disease_or_pest(text: str) -> str:
    """Identify specific disease, pest, or physiological disorder."""
    text_lower = text.lower()
    for key, disease_name in DISEASE_KEYWORDS.items():
        if key in text_lower:
            return disease_name
    return ""


def detect_category(text: str) -> str:
    """Identify agricultural category."""
    text_lower = text.lower()
    for pattern, category in CATEGORY_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return category
    return "General Crop Care"


def detect_treatment_type(text: str) -> str:
    """Determine whether the passage discusses organic, chemical, or cultural remedies."""
    text_lower = text.lower()
    is_organic = bool(re.search(r"\b(neem|bio|trichoderma|pseudomonas|organic|botanical|cow\s*dung|manure|compost|solarization)\b", text_lower))
    is_chemical = bool(re.search(r"\b(mancozeb|copper|chlorothalonil|carbendazim|imidacloprid|dimethoate|chemical|fungicide|insecticide|spray\s*at|ml/l|g/l)\b", text_lower))
    is_cultural = bool(re.search(r"\b(rotation|spacing|sanitation|weeding|resistant\s*varieties|certified\s*seed|drip|pruning)\b", text_lower))

    types = []
    if is_organic:
        types.append("Organic / Bio-Control")
    if is_chemical:
        types.append("Chemical Protection")
    if is_cultural:
        types.append("Cultural & Preventive")

    return " & ".join(types) if types else "General Management"


def extract_canonical_metadata(
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    title: Optional[str] = None,
    text_content: Optional[str] = None,
    page: int = 1,
    doc_id: int = 0,
    version: int = 1,
    chunk_index: int = 0,
    existing_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extracts and standardizes canonical agricultural metadata for LeafLine RAG vector chunks.
    Tags each chunk with crop, category, disease/pest, treatment type, source file, and page.
    """
    existing = existing_meta or {}
    file_str = str(file_path or existing.get("source", ""))
    filename = Path(file_str).name if file_str else ""
    content = text_content or ""
    title_str = str(title or existing.get("title", filename or "LeafLine Agritech Document"))

    # Extract crop
    crop = existing.get("crop")
    if not crop:
        crop = detect_crop(content[:1500], filename)

    # Extract category
    category = existing.get("category")
    if not category:
        category = detect_category(content)

    # Extract disease / pest
    disease_or_pest = existing.get("disease_or_pest")
    if not disease_or_pest:
        disease_or_pest = detect_disease_or_pest(content)

    # Extract treatment type
    treatment_type = existing.get("treatment_type")
    if not treatment_type:
        treatment_type = detect_treatment_type(content)

    canonical: Dict[str, Any] = {
        "crop": crop,
        "category": category,
        "disease_or_pest": disease_or_pest,
        "treatment_type": treatment_type,
        "title": title_str,
        "source": file_str,
        "filename": filename,
        "page": page or existing.get("page", 1),
        "chunk_index": chunk_index or existing.get("chunk_index", 0),
        "doc_id": doc_id or existing.get("doc_id", 0),
        "version": version or existing.get("version", 1),
    }

    return canonical
