from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Dict, Any

from app.schemas.disease import (
    PredictionResponse,
    SupportedCropsResponse,
    CropClassesResponse,
    CropInfo,
)
from app.services.model_service import get_supported_crops, get_crop_classes
from app.services.prediction_service import analyze_leaf_image
from app.config import settings

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
    "/analyze",
    response_model=PredictionResponse,
    summary="Analyze Leaf Image for Disease",
    description=(
        "Upload a leaf photograph and crop type. The service selects the crop-specific classifier, "
        "predicts possible diseases, returns real confidence information, and provides verified ICAR management guidelines."
    ),
)
async def analyze_leaf(
    crop: str = Form(..., description="Target crop name (e.g. tomato, chilli, brinjal, okra, cucumber)"),
    image: UploadFile = File(..., description="Leaf photograph (JPEG, PNG, WebP)"),
):
    # Validate crop is supported
    classes_info = get_crop_classes(crop)
    if not classes_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported crop '{crop}'. Supported crops: tomato, chilli, brinjal, okra, cucumber.",
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
    prediction = analyze_leaf_image(crop, content, max_size_bytes=max_size_bytes)
    return prediction
