"""
LeafLine AI - Gemini Vision-Based Plant Disease Detection Module
---------------------------------------------------------------
Integrates Google's official `google-genai` SDK for multimodal leaf pathology diagnosis.
Enforces 5 MB file size limit, image validation/resizing, structured schema generation,
and robust error and quota handling.
"""

import os
import io
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
for env_path in [_PROJECT_ROOT / ".env", _CURRENT_DIR / ".env", Path(".env")]:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

logger = logging.getLogger(__name__)

# Constants
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB maximum image size
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()


class PlantAnalysisSchema(BaseModel):
    """Pydantic schema for structured Gemini Vision response."""
    is_plant_visible: bool = Field(
        description="True if a plant, leaf, crop, fruit, or agricultural specimen is visible in the photograph, False if non-plant, person, object, or room."
    )
    crop: str = Field(
        description="Identified crop or plant common name (e.g., Tomato, Chilli, Brinjal, Potato, Cucumber, Beans, Okra, Spinach) or 'Unknown' or 'None'."
    )
    disease_name: str = Field(
        description="Suspected condition or disease name (e.g., Early Blight, Powdery Mildew, Healthy, Bacterial Spot) or 'Unknown' or 'Inconclusive'."
    )
    visible_symptoms: List[str] = Field(
        description="Distinct visual physical symptoms directly observable on the foliage (e.g., concentric brown spots with yellow halos, leaf curling, marginal necrosis)."
    )
    assessment: str = Field(
        description="Preliminary visual assessment explaining the suspected condition. Must explicitly indicate this is a visual observation, not a confirmed laboratory diagnosis."
    )
    uncertainty: str = Field(
        description="Explanation of diagnostic uncertainty, limitations of visual inspection, ambiguous patterns, or overlapping environmental/nutritional stressors."
    )
    reasoning: str = Field(
        description="Objective step-by-step pathological reasoning connecting visible symptoms to the suspected disease while distinguishing symptoms from disease causation."
    )


PROMPT_GEMINI_VISION = """You are a senior plant pathologist and agricultural diagnostician.
Analyze this photograph to provide a preliminary visual assessment of plant condition.

Strict Diagnostic Rules:
1. Check if a plant leaf, crop, fruit, or agricultural subject is visible:
   - If this image does NOT contain a plant (e.g. human portrait, animal, vehicle, indoor furniture, tool, paper document, or non-plant object):
     Set "is_plant_visible": false
     Set "crop": "None"
     Set "disease_name": "Non-Plant Subject"
     Set "visible_symptoms": []
     Set "assessment": "No plant or agricultural specimen is visible in the provided image."
     Set "uncertainty": "High"
     Set "reasoning": "The photograph does not depict plant foliage or crops."

2. If a plant is visible:
   - Set "is_plant_visible": true
   - Identify the crop if recognizable (e.g. Tomato, Chilli, Brinjal, Potato, Beans, Cucumber, Okra, Spinach). If uncertain, write "Unknown".
   - Distinguish carefully between VISIBLE SYMPTOMS (e.g., dark brown concentric spots, yellow halos, leaf cupping, wilting) and the SUSPECTED DISEASE (e.g., Early Blight, Healthy).
   - If the foliage appears normal, vibrant, and free of lesions, set "disease_name": "Healthy".
   - If symptoms are ambiguous, blurry, or cannot be reliably determined from visual patterns alone, set "disease_name": "Inconclusive" or "Unknown".
   - Do NOT claim that visual analysis is a confirmed definitive diagnosis.
   - Do NOT invent artificial confidence percentages or fabricate laboratory certainty.
   - Detail the diagnostic limitations and uncertainty in the "uncertainty" field (e.g. noting that fungal blights can visually mimic nutrient deficiencies without microscopic or PCR testing).

Output strictly valid JSON conforming to the schema."""


def get_api_key() -> str:
    """Retrieve GEMINI_API_KEY from environment."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    return key.strip()


def validate_and_preprocess_image(image_bytes: bytes, filename: Optional[str] = None) -> tuple[bool, str, Optional[bytes], str]:
    """
    Validates uploaded image bytes against size and format constraints,
    converts to RGB, and resizes appropriately.

    Returns:
        (is_valid, error_message, processed_bytes, mime_type)
    """
    if not image_bytes:
        return False, "No image data provided.", None, ""

    # Check 5 MB limit
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        size_mb = len(image_bytes) / (1024 * 1024)
        return False, f"Image file exceeds the 5 MB limit (current size: {size_mb:.2f} MB). Please upload a smaller image.", None, ""

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        orig_format = (pil_img.format or "").upper()
        
        # Verify valid image format
        if orig_format not in {"JPEG", "JPG", "PNG", "WEBP", "MPO"}:
            return False, f"Unsupported image format: {orig_format}. Supported formats are JPEG, PNG, and WebP.", None, ""

        # Convert to RGB (handles RGBA, Palette, Grayscale)
        rgb_img = pil_img.convert("RGB")

        # Resize if dimensions exceed 1024x1024 to optimize latency/tokens while preserving pathology detail
        max_dim = 1024
        if rgb_img.width > max_dim or rgb_img.height > max_dim:
            rgb_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        # Re-encode to high quality JPEG
        out_buf = io.BytesIO()
        rgb_img.save(out_buf, format="JPEG", quality=90, optimize=True)
        processed_bytes = out_buf.getvalue()

        return True, "", processed_bytes, "image/jpeg"

    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return False, "Malformed or unreadable image file. Please upload a standard JPG or PNG photograph.", None, ""


def analyze_plant_disease(
    image_bytes: bytes,
    user_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sends the plant image to Gemini Vision API using google-genai SDK,
    enforcing structured JSON extraction, error handling, and model fallbacks.
    """
    api_key = get_api_key()
    if not api_key or api_key == "your_gemini_api_key_here":
        return {
            "success": False,
            "error_type": "config_error",
            "message": "Gemini API key is not configured. Please set GEMINI_API_KEY in your environment or .env file."
        }

    # 1. Validate and preprocess image
    is_valid, err_msg, processed_bytes, mime_type = validate_and_preprocess_image(image_bytes)
    if not is_valid or processed_bytes is None:
        return {
            "success": False,
            "error_type": "validation_error",
            "message": err_msg
        }

    # 2. Initialize official google-genai SDK
    try:
        from google import genai
        from google.genai import types
        from google.genai import errors
    except ImportError:
        logger.error("google-genai package is not installed.")
        return {
            "success": False,
            "error_type": "dependency_error",
            "message": "The official google-genai SDK is not installed. Please run `pip install google-genai`."
        }

    client = genai.Client(api_key=api_key)

    prompt = PROMPT_GEMINI_VISION
    if user_query:
        prompt += f"\n\nFarmer Observation/Question: {user_query.strip()}"

    # Model candidates list for fallback in case of rate-limiting or specific model quota exhaustion
    model_candidates = [DEFAULT_MODEL]
    for fallback_m in ["gemini-2.5-flash-lite", "gemini-flash-latest", "gemini-2.5-pro", "gemini-2.5-flash"]:
        if fallback_m not in model_candidates:
            model_candidates.append(fallback_m)

    response_text = None
    used_model = None
    last_error = None

    image_part = types.Part.from_bytes(data=processed_bytes, mime_type=mime_type)

    for model_name in model_candidates:
        try:
            logger.info(f"Invoking Gemini vision model: {model_name}...")
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PlantAnalysisSchema,
                temperature=0.2,
            )
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image_part],
                config=config
            )
            if response and response.text:
                response_text = response.text
                used_model = model_name
                break
        except errors.ClientError as ce:
            err_str = str(ce)
            logger.warning(f"Model {model_name} client error: {err_str[:160]}")
            last_error = ce
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "NOT_FOUND" in err_str or "404" in err_str:
                # Quota exhausted or model deprecated, try next model candidate
                continue
            break
        except Exception as e:
            logger.warning(f"Model {model_name} invocation error: {e}")
            last_error = e
            continue

    if not response_text:
        err_msg = str(last_error) if last_error else "Empty response from Gemini vision service."
        if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
            return {
                "success": False,
                "error_type": "rate_limit",
                "message": "Gemini API rate limit or daily quota reached. Please retry in a few moments or use a key with available quota."
            }
        return {
            "success": False,
            "error_type": "api_error",
            "message": "Failed to communicate with Gemini vision model. Please check network connection and API status."
        }

    # 3. Parse and validate structured output
    try:
        # Strip potential markdown fences if present
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        raw_data = json.loads(clean_text)
        validated = PlantAnalysisSchema(**raw_data)
        data = validated.model_dump()

    except Exception as parse_err:
        logger.warning(f"Structured JSON schema validation fallback: {parse_err}")
        try:
            data = json.loads(clean_text)
            data["is_plant_visible"] = bool(data.get("is_plant_visible", True))
            data["crop"] = str(data.get("crop", "Unknown"))
            data["disease_name"] = str(data.get("disease_name", "Inconclusive"))
            data["visible_symptoms"] = list(data.get("visible_symptoms", []))
            data["assessment"] = str(data.get("assessment", response_text))
            data["uncertainty"] = str(data.get("uncertainty", "High uncertainty due to unstructured output."))
            data["reasoning"] = str(data.get("reasoning", "Extracted from vision description."))
        except Exception:
            return {
                "success": False,
                "error_type": "parse_error",
                "message": "Received invalid response format from vision model."
            }

    return {
        "success": True,
        "model": used_model,
        "is_plant_visible": data["is_plant_visible"],
        "crop": data["crop"],
        "disease_name": data["disease_name"],
        "visible_symptoms": data["visible_symptoms"],
        "assessment": data["assessment"],
        "uncertainty": data["uncertainty"],
        "reasoning": data["reasoning"]
    }
