import re
from typing import List, Dict, Any, Tuple

STOP_WORDS = {
    "what", "is", "are", "the", "explain", "give", "show", "list", "tell", "me",
    "for", "in", "of", "and", "or", "an", "a", "to", "with", "from", "by", "its",
    "it", "this", "that", "these", "those", "how", "why", "which", "plant", "plants",
    "crop", "crops", "leaf", "leaves", "farm", "farmer", "detail", "detailed", "about"
}

CROPS = {
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
    "spinach": "Spinach (Palak)",
    "palak": "Spinach (Palak)"
}


def detect_query_crop(query: str) -> str:
    """Identify if a specific crop was mentioned in the query."""
    q_lower = query.lower()
    for key, crop_name in CROPS.items():
        if re.search(rf"\b{re.escape(key)}\b", q_lower):
            return crop_name
    return ""


def validate_candidate_chunk(
    doc: Any,
    resolved_query: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Validates a retrieved vector document chunk against the agricultural query parameters.
    Returns (accepted: bool, rejection_reason: str).
    """
    meta = getattr(doc, "metadata", {}) or {}
    text = (getattr(doc, "page_content", "") or "").lower()

    query_crop = resolved_query.get("crop", "").strip()
    topic = resolved_query.get("topic", "").lower().strip()

    doc_crop = str(meta.get("crop", "")).strip()

    # 1. CROP CONSISTENCY CHECK:
    # If the user specifically asked about e.g. Tomato, avoid returning chunks strictly labeled for a different specific crop (like Beans or Spinach)
    if query_crop and doc_crop and doc_crop not in ("General Horticulture", "General Crop Care", ""):
        # Check if the crops conflict
        q_first_word = query_crop.split()[0].lower()
        d_first_word = doc_crop.split()[0].lower()
        if q_first_word != d_first_word:
            return False, f"Crop mismatch: requested '{query_crop}', found chunk for '{doc_crop}'"

    # 2. TOPIC RELEVANCE CHECK:
    if topic and len(topic) > 2:
        topic_words = [
            w for w in re.findall(r"\b[a-z0-9_\-]+\b", topic)
            if len(w) > 2 and w not in STOP_WORDS
        ]
        if topic_words:
            combined_chunk = f"{text} {str(meta).lower()}"
            matched_count = sum(1 for tw in topic_words if tw in combined_chunk)
            if matched_count == 0:
                return False, f"Topic mismatch: chunk does not contain topic keywords {topic_words}"

    return True, "Accepted"


def filter_and_validate_candidates(
    candidate_docs: List[Any],
    resolved_query: Dict[str, Any]
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Filter candidate documents through the agricultural relevance gate."""
    accepted_docs = []
    debug_records = []

    for idx, doc in enumerate(candidate_docs):
        meta = getattr(doc, "metadata", {}) or {}
        text = getattr(doc, "page_content", "")

        accepted, reason = validate_candidate_chunk(doc, resolved_query)

        record = {
            "index": idx,
            "source": meta.get("filename", meta.get("source", "Agritech Guide")),
            "crop": meta.get("crop", "General"),
            "category": meta.get("category", ""),
            "disease_or_pest": meta.get("disease_or_pest", ""),
            "page": meta.get("page", 1),
            "accepted": accepted,
            "rejection_reason": reason,
            "snippet": text[:200] + ("..." if len(text) > 200 else "")
        }
        debug_records.append(record)

        if accepted:
            accepted_docs.append(doc)

    # Fallback: If strict crop filtering rejected everything, return top candidates rather than empty
    if not accepted_docs and candidate_docs:
        print("[Relevance Gate] Fallback: returning candidate documents to ensure answer coverage.")
        return candidate_docs, debug_records

    return accepted_docs, debug_records
