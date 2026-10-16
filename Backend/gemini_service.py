import os
import json
import base64
import logging
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Union, Dict, Any, Optional
from PIL import Image
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Search and load .env
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
for env_path in [_PROJECT_ROOT / ".env", _CURRENT_DIR / ".env", Path(".env")]:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break


def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from environment."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    return key.strip()


def get_gemini_model_name() -> str:
    """Retrieve Gemini model name."""
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()


def is_gemini_configured() -> bool:
    """Check if a valid Gemini API key is configured."""
    key = get_gemini_api_key()
    return bool(key and key != "your_gemini_api_key_here" and len(key) > 10)


PROMPT_SYSTEM = """You are an expert agricultural scientist, plant pathologist, and senior agronomist.
Examine this plant photograph closely and provide comprehensive agronomic diagnosis.

If the image does NOT contain a plant, leaf, crop, fruit, or agricultural subject (for example, if it is a human portrait, indoor room, lamp, car, document, or non-plant object):
- Set "is_plant_or_leaf": false
- Set "health_status": "Non-plant"
- Set "crop": "None"
- Set "diagnosis": "Not a plant or leaf image"
- In "summary", politely explain that the uploaded photo does not contain a crop leaf or plant, and ask the user to upload a clear photograph of a crop leaf.
- Keep treatment arrays empty.

If the image DOES contain a plant or crop leaf:
- Set "is_plant_or_leaf": true
- Identify the crop (e.g. Tomato, Chilli, Brinjal, Okra, Cucurbits, Beans, Spinach, etc.)
- Determine "health_status": "Healthy" | "Diseased" | "Pest Infested"
- Provide specific "diagnosis" (e.g. "Early Blight", "Bacterial Wilt", "Powdery Mildew", "Healthy Plant")
- Provide "confidence_level": "High" | "Medium" | "Low"
- Provide "confidence_score": numeric float between 0.0 and 1.0 (e.g. 0.95)
- List observed "visual_symptoms" (lesions, chlorosis, halos, spots, curling, wilting)
- Explain biological "causes" (fungus, bacterium, virus, environmental conditions)
- Detail practical "organic_treatment" options (botanical sprays, Trichoderma, Pseudomonas, neem oil, pruning)
- Detail "chemical_treatment" options with active ingredients and safety warnings (suppress or keep empty if healthy)
- List actionable "preventive_measures" (crop rotation, spacing, drip irrigation, sanitation)
- Provide a helpful, respectful conversational "summary" for the farmer

Return ONLY a valid JSON object matching this exact schema:
{
  "is_plant_or_leaf": true,
  "crop": "Crop Name",
  "health_status": "Healthy",
  "diagnosis": "Condition Name",
  "confidence_level": "High",
  "confidence_score": 0.95,
  "visual_symptoms": ["Symptom 1", "Symptom 2"],
  "causes": "Explanation of cause",
  "organic_treatment": ["Remedy 1"],
  "chemical_treatment": ["Chemical 1"],
  "preventive_measures": ["Prevention 1"],
  "summary": "Conversational summary for the farmer"
}"""


def _prepare_image_bytes(image_input: Union[Image.Image, bytes, str, Path]) -> tuple[bytes, str]:
    """Convert various image input formats to standard JPEG bytes and MIME type."""
    if isinstance(image_input, (str, Path)):
        with open(image_input, "rb") as f:
            data = f.read()
        return data, "image/jpeg"
    elif isinstance(image_input, bytes):
        return image_input, "image/jpeg"
    elif isinstance(image_input, Image.Image):
        buf = BytesIO()
        rgb_img = image_input.convert("RGB")
        rgb_img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")


def analyze_plant_image_with_gemini(
    image_input: Union[Image.Image, bytes, str, Path],
    user_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a plant leaf image using Google Gemini Multimodal Vision.
    Returns structured analysis dictionary.
    """
    if not is_gemini_configured():
        return {
            "success": False,
            "configured": False,
            "error": "GEMINI_API_KEY is not configured",
            "message": "Please add your Gemini API key in the .env file to enable Gemini multimodal AI analysis.",
            "summary": "Gemini API key is missing. Add GEMINI_API_KEY=your_key in the .env file."
        }

    api_key = get_gemini_api_key()
    model_name = get_gemini_model_name()

    try:
        img_bytes, mime_type = _prepare_image_bytes(image_input)
    except Exception as e:
        logger.error(f"Failed to prepare image for Gemini: {e}")
        return {
            "success": False,
            "configured": True,
            "error": f"Invalid image format: {str(e)}",
            "summary": "Could not read the uploaded image."
        }

    prompt_text = PROMPT_SYSTEM
    if user_query:
        prompt_text += f"\n\nFarmer Question/Note: {user_query.strip()}"

    response_text = None

    # 1. Try google.generativeai SDK
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        models_to_try = [model_name, "gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-pro"]
        for candidate in models_to_try:
            try:
                model = genai.GenerativeModel(candidate)
                response = model.generate_content([prompt_text, pil_image])
                if response and response.text:
                    response_text = response.text
                    model_name = candidate
                    break
            except Exception:
                continue
    except Exception as sdk_err:
        logger.warning(f"SDK call notice: {sdk_err}. Trying direct REST endpoint...")

    # 2. Resilient Direct REST Fallback
    if not response_text:
        try:
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt_text},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": b64_img
                                }
                            }
                        ]
                    }
                ]
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=35) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        response_text = parts[0].get("text", "")
        except Exception as rest_err:
            logger.error(f"Gemini REST API error: {rest_err}")
            return {
                "success": False,
                "configured": True,
                "error": f"Gemini API request failed: {str(rest_err)}",
                "summary": "Gemini analysis could not be completed."
            }

    if not response_text:
        return {
            "success": False,
            "configured": True,
            "error": "Empty response from Gemini API",
            "summary": "Gemini returned an empty diagnosis."
        }

    # Clean and parse JSON
    cleaned_json = response_text.strip()
    if cleaned_json.startswith("```json"):
        cleaned_json = cleaned_json[7:]
    elif cleaned_json.startswith("```"):
        cleaned_json = cleaned_json[3:]
    if cleaned_json.endswith("```"):
        cleaned_json = cleaned_json[:-3]
    cleaned_json = cleaned_json.strip()

    try:
        parsed_data = json.loads(cleaned_json)
    except json.JSONDecodeError:
        parsed_data = {
            "is_plant_or_leaf": True,
            "crop": "Agricultural Plant",
            "health_status": "Analyzed",
            "diagnosis": "Condition Analyzed",
            "confidence_level": "Medium",
            "confidence_score": 0.85,
            "visual_symptoms": [],
            "causes": "See detailed summary",
            "organic_treatment": [],
            "chemical_treatment": [],
            "preventive_measures": [],
            "summary": response_text
        }

    return {
        "success": True,
        "configured": True,
        "model": model_name,
        "analysis": parsed_data,
        "raw_text": response_text
    }
