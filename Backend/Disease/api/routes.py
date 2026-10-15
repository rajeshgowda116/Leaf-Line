from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from schemas.disease import (
    PredictionResponse,
    SupportedCropsResponse,
    CropClassesResponse,
    CropDetectionResponse,
    CropInfo,
)
from services.model_service import get_supported_crops, get_crop_classes
from services.prediction_service import analyze_leaf_image
from services.crop_service import detect_crop_type
from services.image_service import validate_and_open_image
from config import settings

router = APIRouter(prefix="/disease", tags=["Disease Detection"])


@router.get(
    "/crops",
    response_model=SupportedCropsResponse,
    summary="List Supported Crops",
    description="Returns all 5 supported agricultural crops with their verified model information.",
)
async def list_crops():
    crop_list = get_supported_crops()
    return SupportedCropsResponse(
        crops=[CropInfo(**c) for c in crop_list]
    )


@router.get(
    "/crops/{crop}/classes",
    response_model=CropClassesResponse,
    summary="Get Disease Classes for a Crop",
    description="Returns the verified condition and disease labels for the specified crop.",
)
async def get_classes(crop: str):
    info = get_crop_classes(crop)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Crop '{crop}' is not supported. Please choose from: tomato, chilli, brinjal, okra, cucumber.",
        )
    return CropClassesResponse(**info)


@router.post(
    "/detect-crop",
    response_model=CropDetectionResponse,
    summary="Detect Crop & Validate Leaf Presence",
    description="Analyzes the image to detect whether it is an agricultural crop leaf and automatically identifies the crop type.",
)
async def detect_crop(
    image: UploadFile = File(..., description="Photograph to analyze (JPEG, PNG, WebP)"),
):
    try:
        content = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image stream: {str(e)}",
        )

    max_size_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    pil_image = validate_and_open_image(content, max_size_bytes=max_size_bytes)
    result = detect_crop_type(pil_image)
    return CropDetectionResponse(**result)


@router.post(
    "/analyze",
    response_model=PredictionResponse,
    summary="Analyze Leaf Image for Disease",
    description=(
        "Upload a leaf photograph. Validates leaf presence (rejecting non-plant objects like diyas, candles, vehicles), "
        "detects or verifies the crop type, predicts diseases, and returns verified ICAR management guidelines."
    ),
)
async def analyze_leaf(
    crop: str = Form("auto", description="Target crop name (e.g. tomato, chilli, brinjal, okra, cucumber) or 'auto' for automatic crop detection"),
    image: UploadFile = File(..., description="Leaf photograph (JPEG, PNG, WebP)"),
):
    clean_crop = crop.strip().lower() if crop else "auto"

    # Validate crop if an explicit crop name was provided
    if clean_crop not in ("auto", "unknown", ""):
        classes_info = get_crop_classes(clean_crop)
        if not classes_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported crop '{crop}'. Supported crops: tomato, chilli, brinjal, okra, cucumber, or 'auto'.",
            )

    # Read bytes
    try:
        content = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image stream: {str(e)}",
        )

    max_size_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    prediction = analyze_leaf_image(clean_crop, content, max_size_bytes=max_size_bytes)
    return prediction
