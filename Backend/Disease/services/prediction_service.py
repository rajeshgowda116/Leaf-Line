from typing import Dict, Any
from PIL import Image
from fastapi import HTTPException, status

from schemas.disease import PredictionResponse, ClassScore
from services.image_service import validate_and_open_image
from services.crop_service import validate_leaf_image, detect_crop_type
from services.model_service import predict_disease
from services.treatment_service import get_treatment_guidance, normalize_crop_name


def analyze_leaf_image(
    crop: str,
    file_bytes: bytes,
    max_size_bytes: int = 10 * 1024 * 1024,
) -> PredictionResponse:
    """
    Coordinates the full end-to-end analysis:
    1. Validates and decodes image bytes
    2. Runs Crop & Leaf Validation Model to reject non-plant objects (diya, candle, vehicles, etc.)
    3. Detects agricultural crop type automatically
    4. Runs model inference for the verified crop
    5. Retrieves verified ICAR treatment guidance
    6. Formulates a complete, structured response with confidence metrics
    """
    # 1. Image validation & EXIF orientation correction
    image: Image.Image = validate_and_open_image(file_bytes, max_size_bytes=max_size_bytes)

    # 2. Crop & Leaf Presence Validation Model
    is_leaf, val_msg, leaf_meta = validate_leaf_image(image)
    if not is_leaf:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"No crop leaf detected in the uploaded image. {val_msg}. "
                "Please upload a clear photograph of an agricultural crop leaf "
                "(such as Chilli, Tomato, Brinjal, Okra, Cucumber, Potato)."
            ),
        )

    # 3. Automatic Crop Detection Model
    crop_detection = detect_crop_type(image)
    detected_crop = crop_detection.get("detected_crop", "unknown")
    crop_conf = crop_detection.get("confidence", 0.0)

    # Normalize or auto-assign crop
    norm_crop = normalize_crop_name(crop) if crop else "auto"

    from services.model_service import CROP_REGISTRY

    # Crop Routing Policy:
    # 1. If user passed 'auto' or empty, use detected crop if supported
    # 2. If user explicitly specified a crop, use that crop as effective_crop
    if norm_crop in ("auto", "unknown", "", "none"):
        if detected_crop in CROP_REGISTRY:
            effective_crop = detected_crop
        else:
            matched = None
            for cand in crop_detection.get("all_crops", []):
                c_name = cand.get("crop", "")
                if c_name in CROP_REGISTRY:
                    matched = c_name
                    break
            if matched:
                effective_crop = matched
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Detected crop '{detected_crop.title()}' ({crop_conf:.0%} confidence) is not currently supported. "
                        f"LeafLine supports: {', '.join(k.title() for k in CROP_REGISTRY.keys())}."
                    ),
                )
    else:
        effective_crop = norm_crop

    if effective_crop not in CROP_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Crop '{effective_crop.title()}' is not supported. "
                f"LeafLine supports: {', '.join(k.title() for k in CROP_REGISTRY.keys())}."
            ),
        )

    # 4. Predict disease using the crop classifier
    predicted_disease, confidence, model_name, pred_status, top_scores = predict_disease(effective_crop, image)

    # Calibrated uncertainty check: if confidence is below 45%, mark inconclusive
    if confidence is not None and confidence < 0.45:
        pred_status = "inconclusive"
        if not ("inconclusive" in predicted_disease.lower() or "uncertain" in predicted_disease.lower()):
            predicted_disease = f"{predicted_disease} (Inconclusive)"

    # 5. Retrieve treatment guidance
    guidance = get_treatment_guidance(effective_crop, predicted_disease)

    # Source references
    source_refs = []
    if guidance.source_url:
        source_refs.append(guidance.source_url)
    source_refs.append(f"Model: {model_name}")
    source_refs.append("Crop Detector: TreeTrack Vision & Chlorophyll Verification Model")

    formatted_scores = [ClassScore(label=s["label"], score=s["score"]) for s in top_scores[:5]]

    return PredictionResponse(
        crop=effective_crop,
        predicted_disease=predicted_disease,
        confidence=confidence,
        status=pred_status,
        model_name=model_name,
        overview=guidance.overview,
        is_leaf=True,
        detected_crop=detected_crop,
        crop_confidence=crop_conf,
        validation_message=val_msg,
        treatment_guidance=guidance,
        source_references=source_refs,
        human_review_required=True,
        top_predictions=formatted_scores,
    )
