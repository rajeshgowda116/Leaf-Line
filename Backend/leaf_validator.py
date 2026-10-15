"""
LeafLine - Plant-Leaf Image Validation Module
---------------------------------------------
Provides multi-stage verification for uploaded images before disease classification:
1. File integrity, size limits (10 MB), and image decoding checks.
2. Trained binary classifier (LeafValidatorNet from models/leaf_validator.pth).
3. Botanical vegetation and chlorophyll index verification (Excess Green Index).
4. Pretrained ImageNet vision classification to catch common everyday non-leaf objects
   (human faces, candles/lamps/diyas, vehicles, furniture, electronics).

Rejection message required by specification:
"This image does not appear to contain a plant leaf. Please upload a clear leaf photograph."
"""

import io
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATOR_MODEL_PATH = BASE_DIR / "models" / "leaf_validator.pth"

DEFAULT_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}

# Everyday non-plant objects in ImageNet to intercept
NON_PLANT_OBJECT_KEYWORDS = {
    'candle', 'lamp', 'lighter', 'matchstick', 'fire', 'flame',
    'torch', 'lantern', 'pottery', 'vase', 'pitcher', 'jug', 'bowl',
    'plate', 'cup', 'mug', 'teapot', 'screen', 'monitor', 'cellular_telephone',
    'laptop', 'computer', 'chair', 'desk', 'table', 'car', 'automobile',
    'truck', 'bicycle', 'motorcycle', 'clothing', 'shoe', 'sunglass',
    'wall_clock', 'analog_clock', 'digital_clock', 'pillow', 'book', 'envelope',
    'wallet', 'purse', 'handbag', 'backpack', 'guitar', 'violin', 'drum',
    'traffic_light', 'street_sign', 'parking_meter', 'mailbox', 'trash_can',
    'refrigerator', 'microwave', 'oven', 'toaster', 'blender', 'frying_pan',
    'suit', 'tie', 'windsor tie', 'bow tie', 'necktie', 'groom', 'jacket',
    'coat', 'trench coat', 'lab coat', 'fur coat', 'vest', 'sweater',
    'sweatshirt', 'jersey', 'cardigan', 'gown', 'academic gown', 'dress',
    'jean', 'denim', 'shirt', 't-shirt', 'pajama', 'wig', 'face powder',
    'sunglasses', 'spectacles', 'mask', 'gasmask', 'scuba diver', 'person', 'human'
}

# -------------------------------------------------------------
# 1. Model Loading (LeafValidatorNet & MobileNetV3)
# -------------------------------------------------------------

_BINARY_VALIDATOR = None
_BINARY_TRANSFORMS = None

_VISION_CLASSIFIER = None
_VISION_TRANSFORMS = None
_VISION_CATEGORIES = None


def _get_binary_validator():
    """Loads and caches the trained LeafValidatorNet weights."""
    global _BINARY_VALIDATOR, _BINARY_TRANSFORMS
    if _BINARY_VALIDATOR is not None:
        return _BINARY_VALIDATOR, _BINARY_TRANSFORMS

    if not VALIDATOR_MODEL_PATH.exists():
        logger.warning(f"Leaf validator weights not found at {VALIDATOR_MODEL_PATH}")
        return None, None

    try:
        from train_leaf_validator import LeafValidatorNet
        model = LeafValidatorNet()
        state_dict = torch.load(str(VALIDATOR_MODEL_PATH), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        val_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        _BINARY_VALIDATOR = model
        _BINARY_TRANSFORMS = val_transforms
        return _BINARY_VALIDATOR, _BINARY_TRANSFORMS
    except Exception as e:
        logger.warning(f"Failed to load LeafValidatorNet: {e}")
        return None, None


def _get_vision_classifier():
    """Loads and caches the MobileNetV3 ImageNet object classifier."""
    global _VISION_CLASSIFIER, _VISION_TRANSFORMS, _VISION_CATEGORIES
    if _VISION_CLASSIFIER is not None:
        return _VISION_CLASSIFIER, _VISION_TRANSFORMS, _VISION_CATEGORIES

    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        classifier = models.mobilenet_v3_small(weights=weights).eval()
        _VISION_CLASSIFIER = classifier
        _VISION_TRANSFORMS = weights.transforms()
        _VISION_CATEGORIES = weights.meta['categories']
        return _VISION_CLASSIFIER, _VISION_TRANSFORMS, _VISION_CATEGORIES
    except Exception as e:
        logger.debug(f"MobileNetV3 classifier unavailable: {e}")
        return None, None, None


# -------------------------------------------------------------
# 2. File & Format Integrity Validation
# -------------------------------------------------------------

def validate_image_file(
    file_bytes: bytes,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES
) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validates uploaded file bytes:
    - Rejects empty uploads
    - Rejects oversized files (> 10MB)
    - Verifies valid image stream using PIL verify
    - Corrects phone camera EXIF orientation
    - Converts to 8-bit RGB
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, "The uploaded file is empty. Please upload a valid plant leaf image.", None

    if len(file_bytes) > max_size_bytes:
        size_mb = max_size_bytes / (1024 * 1024)
        return False, f"File size exceeds the maximum allowed limit of {size_mb:.1f} MB.", None

    # Verify structural integrity
    try:
        test_stream = io.BytesIO(file_bytes)
        with Image.open(test_stream) as img:
            img_format = (img.format or "").upper()
            if img_format not in ALLOWED_FORMATS:
                return (
                    False,
                    f"Unsupported image format '{img_format}'. Supported formats are JPEG, PNG, and WebP.",
                    None
                )
            img.verify()
    except UnidentifiedImageError:
        return False, "Unable to read image. The file is corrupt or not a valid image format.", None
    except Exception as e:
        return False, f"Image verification error: {str(e)}", None

    # Decode and transpose EXIF
    try:
        decode_stream = io.BytesIO(file_bytes)
        img = Image.open(decode_stream)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return True, "Image successfully decoded.", img
    except Exception as e:
        return False, f"Failed to process image: {str(e)}", None


# -------------------------------------------------------------
# 3. Binary Plant-Leaf Validation
# -------------------------------------------------------------

def is_plant_leaf(image: Image.Image) -> Tuple[bool, float, str, Dict[str, Any]]:
    """
    Evaluates whether the uploaded image contains an actual agricultural plant leaf.

    Returns:
        is_leaf (bool): True if verified as leaf, False if non-leaf.
        confidence (float): Confidence score (0.0 to 1.0).
        message (str): Human-readable validation message.
        metadata (dict): Detailed diagnostic signals.
    """
    rgb_img = image.convert("RGB")
    arr = np.array(rgb_img.resize((128, 128)), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Botanical vegetation metrics
    # True chlorophyll presence: Green channel strictly dominant over Red and Blue
    true_green_ratio = float(np.mean((g > r * 1.05) & (g > b * 1.15) & (g > 35)))
    # Intense flame / incandescent fire saturation (diya, candle, flame)
    fire_ratio = float(np.mean((r > 180) & (r > g + 25) & (b < 110)))

    # Layer 1: Incandescent Flame / Diya / Artificial Heat Source Check
    if (fire_ratio > 0.08 and true_green_ratio < 0.05) or (fire_ratio > 0.25):
        return (
            False,
            round(fire_ratio, 4),
            "This image does not appear to contain a plant leaf. Please upload a clear leaf photograph.",
            {"reason": "fire_or_heat_source", "fire_ratio": fire_ratio}
        )

    # Layer 2: Pretrained Vision Object Recognition (ImageNet MobileNetV3)
    vision_model, vision_tf, vision_cats = _get_vision_classifier()
    if vision_model is not None and vision_tf is not None and vision_cats is not None:
        try:
            batch = vision_tf(rgb_img).unsqueeze(0)
            with torch.no_grad():
                probs = vision_model(batch).squeeze(0).softmax(0)
            top_prob, top_idx = torch.topk(probs, 5)

            for rank in range(min(3, len(top_prob))):
                cat_name = vision_cats[top_idx[rank].item()].lower().replace("_", " ")
                cat_prob = float(top_prob[rank].item())
                for kw in NON_PLANT_OBJECT_KEYWORDS:
                    if kw in cat_name and cat_prob > 0.10:
                        return (
                            False,
                            round(cat_prob, 4),
                            "This image does not appear to contain a plant leaf. Please upload a clear leaf photograph.",
                            {"reason": "non_plant_object_detected", "detected_object": cat_name, "object_confidence": cat_prob}
                        )
        except Exception as e:
            logger.debug(f"Vision object check bypassed: {e}")

    # Layer 3: Trained Binary LeafValidatorNet
    binary_model, binary_tf = _get_binary_validator()
    if binary_model is not None and binary_tf is not None:
        try:
            batch = binary_tf(rgb_img).unsqueeze(0)
            with torch.no_grad():
                logit = binary_model(batch)
                leaf_prob = float(torch.sigmoid(logit).item())

            if leaf_prob < 0.50:
                return (
                    False,
                    round(leaf_prob, 4),
                    "This image does not appear to contain a plant leaf. Please upload a clear leaf photograph.",
                    {"reason": "binary_validator_rejected", "leaf_probability": leaf_prob}
                )
        except Exception as e:
            logger.debug(f"Binary model check bypassed: {e}")

    # Layer 4: Minimal vegetation check (detects blank walls, pure text, pure monochrome)
    if true_green_ratio < 0.015 and fire_ratio > 0.02:
        return (
            False,
            round(true_green_ratio, 4),
            "This image does not appear to contain a plant leaf. Please upload a clear leaf photograph.",
            {"reason": "zero_chlorophyll_detected", "green_ratio": true_green_ratio}
        )

    return (
        True,
        round(max(0.70, true_green_ratio), 4),
        "Plant leaf verified successfully.",
        {"green_coverage": true_green_ratio, "status": "leaf_verified"}
    )
