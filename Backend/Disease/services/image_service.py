import io
from typing import Tuple
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
import torchvision.transforms as transforms
from fastapi import HTTPException, status

ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
DEFAULT_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Standard ImageNet / ConvNeXt normalization
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def validate_and_open_image(
    file_bytes: bytes,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
) -> Image.Image:
    """
    Validates uploaded image file bytes and converts to an orientation-corrected RGB PIL Image.

    Checks:
    - Non-empty byte content
    - File size limit
    - Valid image header and integrity (using PIL verify)
    - Supported formats (JPEG, PNG, WebP)
    - Auto-corrects EXIF orientation from smartphone cameras
    - Converts to 8-bit RGB
    """
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty. Please provide a valid leaf image.",
        )

    if len(file_bytes) > max_size_bytes:
        size_mb = max_size_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"Image size exceeds the maximum permitted limit of {size_mb:.1f} MB.",
        )

    # First pass: verify structure
    try:
        raw_stream = io.BytesIO(file_bytes)
        with Image.open(raw_stream) as test_img:
            image_format = (test_img.format or "").upper()
            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=(
                        f"Unsupported image format '{image_format}'. "
                        f"Please upload an image in JPEG, PNG, or WebP format."
                    ),
                )
            test_img.verify()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to identify image. The file appears to be corrupted or not a valid image.",
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image verification failed: {str(e)}",
        )

    # Second pass: open, correct EXIF orientation, convert to RGB
    try:
        stream = io.BytesIO(file_bytes)
        img = Image.open(stream)
        # Handle orientation tag from phone cameras
        img = ImageOps.exif_transpose(img)
        # Convert any mode (RGBA, CMYK, P, L) to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process and decode image: {str(e)}",
        )


def preprocess_for_model(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Applies standard evaluation preprocessing for ConvNeXt models:
    - Resize to target size (224, 224)
    - Convert to Float Tensor [0, 1]
    - Normalize with ImageNet mean & standard deviation
    - Adds batch dimension: (1, 3, H, W)
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    tensor = transform_pipeline(image)
    return tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)
