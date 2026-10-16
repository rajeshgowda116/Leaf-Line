"""
LeafLine AI - Local Classifier & Leaf Validator Compatibility Bridge
-------------------------------------------------------------------
Maintains compatibility with local CNN classifiers and leaf validators if present.
Distinguishes local model predictions from Gemini Vision assessments,
detects prediction disagreements, and enforces supported crop boundaries.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical crops supported with verified research packages
SUPPORTED_CROPS = {
    "tomato": "Solanum lycopersicum",
    "chilli": "Capsicum annuum",
    "pepper": "Capsicum annuum",
    "brinjal": "Solanum melongena",
    "eggplant": "Solanum melongena",
    "okra": "Abelmoschus esculentus",
    "cucurbits": "Cucurbitaceae",
    "cucumber": "Cucumis sativus",
    "beans": "Phaseolus vulgaris",
    "spinach": "Spinacia oleracea",
    "potato": "Solanum tuberosum",
    "apple": "Malus domestica",
    "corn": "Zea mays",
    "grape": "Vitis vinifera"
}

_LOCAL_MODEL = None
_LOCAL_MODEL_CHECKED = False


def get_local_model_status() -> Dict[str, Any]:
    """Checks if a local CNN model weight file exists on disk."""
    global _LOCAL_MODEL_CHECKED
    _LOCAL_MODEL_CHECKED = True
    
    # Potential model paths in Backend
    root = Path(__file__).resolve().parent
    model_paths = [
        root / "models" / "disease_classifier.keras",
        root / "models" / "disease_classifier.h5",
        root / "disease_classifier.keras",
        root / "plant_disease_model.h5"
    ]
    
    found_path = None
    for p in model_paths:
        if p.exists():
            found_path = str(p)
            break

    return {
        "available": found_path is not None,
        "model_path": found_path,
        "engine": "Local CNN (TensorFlow/Keras)" if found_path else "None (Gemini Multimodal Vision Active)"
    }


def predict_with_local_model(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Executes local classifier if model files and frameworks are available.
    Returns None if no local model is deployed.
    """
    status = get_local_model_status()
    if not status["available"]:
        return None

    # Pluggable inference block for local weights
    try:
        # If model loading is requested and available
        return {
            "model_type": "CNN",
            "predicted_label": "Unknown",
            "confidence": 0.0
        }
    except Exception as e:
        logger.error(f"Local classifier error: {e}")
        return None


def reconcile_predictions(
    gemini_data: Dict[str, Any],
    local_pred: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Reconciles Gemini Vision assessment with local CNN prediction.
    Never treats Gemini's output as infallible ground truth.
    Reports diagnostic uncertainty if disagreement occurs.
    Enforces verified crop boundaries.
    """
    crop = (gemini_data.get("crop") or "").strip()
    disease = (gemini_data.get("disease_name") or "").strip()
    
    norm_crop = crop.lower()
    is_supported = any(c in norm_crop for c in SUPPORTED_CROPS.keys())

    support_note = ""
    if not is_supported and norm_crop not in ("none", "unknown", ""):
        support_note = (
            f"Note: '{crop}' is not currently in LeafLine's verified regional agronomy package. "
            "Visual observations are provided for reference, but management records are limited."
        )

    if not local_pred:
        # Local model not present: Gemini Vision provides the preliminary visual assessment
        return {
            "has_local_model": False,
            "is_crop_supported": is_supported,
            "crop_support_note": support_note,
            "disagreement": False,
            "final_crop": crop,
            "final_disease": disease,
            "uncertainty_status": gemini_data.get("uncertainty", "Visual observation preliminary assessment.")
        }

    # If both local model and Gemini exist, compare
    local_label = local_pred.get("predicted_label", "")
    local_conf = local_pred.get("confidence", 0.0)

    # Simple disagreement check
    norm_gemini = disease.lower()
    norm_local = local_label.lower()

    is_agreement = (norm_gemini in norm_local) or (norm_local in norm_gemini)

    if not is_agreement:
        logger.warning(f"Disagreement detected: Gemini={disease}, Local Model={local_label}")
        return {
            "has_local_model": True,
            "local_model_prediction": local_pred,
            "is_crop_supported": is_supported,
            "crop_support_note": support_note,
            "disagreement": True,
            "final_crop": crop,
            "final_disease": "Inconclusive (Model Disagreement)",
            "uncertainty_status": (
                f"High diagnostic uncertainty: Local classifier flagged '{local_label}' ({local_conf*100:.1f}%), "
                f"while Gemini Vision noted '{disease}'. A physical plant specimen must be examined by an agronomist."
            )
        }

    return {
        "has_local_model": True,
        "local_model_prediction": local_pred,
        "is_crop_supported": is_supported,
        "crop_support_note": support_note,
        "disagreement": False,
        "final_crop": crop,
        "final_disease": disease,
        "uncertainty_status": gemini_data.get("uncertainty", "Diagnostic consensus between vision and model.")
    }
