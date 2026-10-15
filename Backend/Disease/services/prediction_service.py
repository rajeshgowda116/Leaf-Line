from typing import Dict, Any
from PIL import Image

from schemas.disease import PredictionResponse, ClassScore
from services.image_service import validate_and_open_image
from services.model_service import predict_disease
from services.treatment_service import get_treatment_guidance, normalize_crop_name


def analyze_leaf_image(
    crop: str,
    file_bytes: bytes,
    max_size_bytes: int = 10 * 1024 * 1024,
) -> PredictionResponse:
    """
    Coordinates the full end-to-end analysis:
    1. Validates and opens image bytes
    2. Runs model inference for the crop
    3. Retrieves verified ICAR treatment guidance
    4. Formulates a complete, structured response with confidence metrics
    """
    norm_crop = normalize_crop_name(crop)

    # 1. Image validation
    image: Image.Image = validate_and_open_image(file_bytes, max_size_bytes=max_size_bytes)

    # 2. Predict disease
    predicted_disease, confidence, model_name, status, top_scores = predict_disease(norm_crop, image)

    # Check for inconclusive result if evidence is ambiguous or confidence is exceptionally low
    if confidence is not None and confidence < 0.20:
        status = "inconclusive"

    # 3. Retrieve treatment guidance
    guidance = get_treatment_guidance(norm_crop, predicted_disease)

    # Source references
    source_refs = []
    if guidance.source_url:
        source_refs.append(guidance.source_url)
    source_refs.append(f"Model: {model_name} (Hugging Face Hub)")

    formatted_scores = [ClassScore(label=s["label"], score=s["score"]) for s in top_scores[:5]]

    return PredictionResponse(
        crop=norm_crop,
        predicted_disease=predicted_disease,
        confidence=confidence,
        status=status,
        model_name=model_name,
        overview=guidance.overview,
        treatment_guidance=guidance,
        source_references=source_refs,
        human_review_required=True,
        top_predictions=formatted_scores,
    )
