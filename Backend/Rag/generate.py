import os
import re
import json
import urllib.request
from typing import Generator, Optional, Any
from dotenv import load_dotenv

load_dotenv()


class GeminiAgronomistLLM:
    """
    High-reliability Gemini LLM wrapper that uses google.generativeai SDK
    with automatic REST fallback to prevent gRPC disconnection errors on Windows.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = ""):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

    def invoke(self, prompt: str) -> Any:
        content = self.generate(prompt)
        # Create an object with .content attribute for LangChain compatibility
        class ResponseWrapper:
            def __init__(self, text):
                self.content = text
            def __str__(self):
                return self.content
        return ResponseWrapper(content)

    def generate(self, prompt: str) -> str:
        prompt_str = prompt if isinstance(prompt, str) else str(prompt)

        # 1. Try google.generativeai SDK
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            resp = model.generate_content(prompt_str)
            if resp and resp.text:
                return resp.text.strip()
        except Exception as sdk_err:
            print(f"[Gemini SDK Notice] Retrying with REST endpoint: {sdk_err}")

        # 2. Resilient Direct REST Fallback (immune to gRPC drops)
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt_str}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=40) as response:
                result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
        except Exception as rest_err:
            print(f"[Gemini REST Error] {rest_err}")
            raise RuntimeError(f"Gemini API request failed: {rest_err}")

        return "No response received from Gemini."


_llm_instance = None


def get_llm(model_override: str = None):
    """
    Singleton lazy loader for the LLM instance.
    Prioritizes Gemini 2.5 Flash with resilient dual SDK/REST architecture.
    """
    global _llm_instance
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if model_override and api_key:
        return GeminiAgronomistLLM(model_name=model_override, api_key=api_key.strip())

    if _llm_instance is not None:
        return _llm_instance

    if api_key and api_key.strip():
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        _llm_instance = GeminiAgronomistLLM(model_name=model_name, api_key=api_key.strip())
        print(f"[INIT] LeafLine Resilient Gemini LLM initialized ({model_name}).")
        return _llm_instance

    # Fallback to Groq if configured
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and groq_api_key.strip():
        try:
            from langchain_groq import ChatGroq
            _llm_instance = ChatGroq(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                groq_api_key=groq_api_key.strip(),
                temperature=0.2
            )
            print("[INIT] Groq LLM initialized.")
            return _llm_instance
        except Exception:
            pass

    return None


LEAFLINE_AGRONOMIST_SYSTEM_RULES = """
============================================================
STRICT AGRICULTURAL GROUNDING & FARMER ADVISORY RULES:
1. You are LeafLine Senior Agronomist and Plant Health Advisor.
2. The provided CONTEXT FROM AGRICULTURAL RESEARCH GUIDES is your primary source of agronomic truth.
3. Base all diagnosis, symptoms, dosages, and cultural practices strictly on the provided context.
4. Do NOT hallucinate toxic pesticide dosages or unverified chemical combinations.
5. If the context does not specify a dosage or remedy, explicitly state:
   "The available agronomic references do not specify this detail; please consult your local Krishi Vigyan Kendra (KVK) or agricultural extension officer."
6. Always prioritize eco-friendly, organic, biological (Trichoderma, Pseudomonas, neem formulations), and preventive cultural management before chemical sprays.
7. For chemical recommendations, state active ingredients clearly, mention safety gear (gloves, masks), and advise observing the pre-harvest waiting interval (PHI).
============================================================
"""


def build_prompt(question: str, context: str, crop: str = "General") -> str:
    """
    LeafLine Agronomist prompt builder tailored to farmer and horticultural queries:
    - Disease Diagnosis & Symptoms
    - Practical Treatment Plans (Organic & Chemical)
    - Cultural & Preventive Practices
    - Crop Cultivation & Package of Practices
    """
    q_lower = question.lower().strip()
    crop_name = crop.strip() if crop and crop.strip() not in ("General", "All", "General Horticulture") else ""

    if crop_name:
        full_instruction = f"Provide expert agronomic guidance for {crop_name} based on the verified agricultural documents: {question}"
    else:
        full_instruction = f"Provide expert agronomic guidance based on the verified agricultural documents: {question}"

    # Detect Query Intent Modes
    is_diagnosis_mode = bool(re.search(
        r"\b(symptom|symptoms|identify|diagnose|signs?|leaf\s*spot|yellowing|wilting|curling|rot|blight|mildew|what\s+disease|what\s+is\s+wrong)\b",
        q_lower
    ))

    is_treatment_mode = bool(re.search(
        r"\b(treatment|treat|cure|control|remedy|remedies|fungicide|insecticide|spray|medicine|how\s+to\s+manage|management|dosage|how\s+much\s+to\s+spray|neem\s*oil)\b",
        q_lower
    ))

    is_prevention_mode = bool(re.search(
        r"\b(prevent|prevention|avoid|stop|cultural|rotation|spacing|soil\s*solarization|seed\s*treatment|resistant|sanitation)\b",
        q_lower
    ))

    is_cultivation_mode = bool(re.search(
        r"\b(cultivation|how\s+to\s+grow|season|soil|sowing|nursery|transplanting|fertilizer|npk|manure|irrigation|harvest|yield|package\s+of\s+practices)\b",
        q_lower
    ))

    # 1. TREATMENT & REMEDIES MODE
    if is_treatment_mode:
        return f"""You are LeafLine Senior Agronomist and Crop Doctor.
{LEAFLINE_AGRONOMIST_SYSTEM_RULES}

FARMER QUESTION: {full_instruction}
CROP SUBJECT: {crop_name if crop_name else 'Agricultural Crop'}

CRITICAL INSTRUCTIONS:
Structure your response clearly using these sections:
1. ## 🌿 Overview & Immediate Action
   - Concise summary of the condition and immediate response required.
2. ## 🍃 Organic & Bio-Control Remedies (First Line of Defense)
   - Botanical sprays (e.g. neem oil, garlic extract), bio-fungicides (Trichoderma, Pseudomonas), and natural methods supported by context.
3. ## 🧪 Reviewed Chemical Options (Use Responsibly)
   - Recommended chemical fungicides/insecticides with active ingredients, application methods, and strict safety guidelines.
4. ## 🛡️ Preventive & Field Management Practices
   - Water management, pruning infected leaves, sanitation, and crop hygiene.
5. End with a helpful safety advisory notice.

CONTEXT FROM AGRICULTURAL GUIDES:
{context}

EXPERT AGRONOMIST TREATMENT PLAN:
"""

    # 2. DISEASE DIAGNOSIS & SYMPTOMS MODE
    if is_diagnosis_mode:
        return f"""You are LeafLine Senior Agronomist and Plant Pathologist.
{LEAFLINE_AGRONOMIST_SYSTEM_RULES}

FARMER QUESTION: {full_instruction}
CROP SUBJECT: {crop_name if crop_name else 'Agricultural Crop'}

CRITICAL INSTRUCTIONS:
Structure your response clearly using these sections:
1. ## 🔍 Diagnosis & Causal Agent
   - Disease or pest identity, pathogen type (fungus, bacterium, virus, or pest).
2. ## 🌿 Visual Symptoms Observed
   - Clear bullet points describing leaf, stem, flower, or fruit symptoms (spots, halos, chlorosis, wilting, lesions).
3. ## 🔬 Favorable Conditions & Spread
   - Weather, humidity, temperature, or vectors contributing to the disease spread.
4. ## ⚡ Recommended Next Steps
   - Immediate practical actions the grower should take right away.

CONTEXT FROM AGRICULTURAL GUIDES:
{context}

EXPERT PLANT PATHOLOGY DIAGNOSIS:
"""

    # 3. PREVENTIVE & CULTURAL PRACTICES MODE
    if is_prevention_mode:
        return f"""You are LeafLine Senior Agronomist and Crop Health Specialist.
{LEAFLINE_AGRONOMIST_SYSTEM_RULES}

FARMER QUESTION: {full_instruction}
CROP SUBJECT: {crop_name if crop_name else 'Agricultural Crop'}

CRITICAL INSTRUCTIONS:
Provide a step-by-step proactive preventive schedule:
1. ## 🛡️ Pre-Sowing & Seed Treatment
   - Certified seeds, hot water treatment, or bio-agent seed coating.
2. ## 🌱 Field Preparation & Cultural Practices
   - Proper spacing, raised beds, drip irrigation, avoiding waterlogging.
3. ## 🔄 Crop Hygiene & Crop Rotation
   - Rotating non-host crops, weed removal, solarization, and field sanitation.
4. ## 📋 Monitoring & Early Detection
   - Sticky traps, pheromone traps, and visual scouting routines.

CONTEXT FROM AGRICULTURAL GUIDES:
{context}

PROACTIVE PREVENTIVE CARE GUIDE:
"""

    # 4. CULTIVATION & PACKAGE OF PRACTICES MODE
    if is_cultivation_mode:
        return f"""You are LeafLine Senior Horticultural Advisor.
{LEAFLINE_AGRONOMIST_SYSTEM_RULES}

FARMER QUESTION: {full_instruction}
CROP SUBJECT: {crop_name if crop_name else 'Agricultural Crop'}

CRITICAL INSTRUCTIONS:
Provide a comprehensive Package of Practices:
1. ## 🌾 Climate & Soil Requirements
2. ## 📅 Sowing Season & Seed Rate
3. ## 🌿 Nursery & Transplanting Practices
4. ## 💧 Irrigation & Nutrient Management (NPK / Organic Manure)
5. ## 🧺 Harvesting & Expected Yield

CONTEXT FROM AGRICULTURAL GUIDES:
{context}

PACKAGE OF PRACTICES GUIDE:
"""

    # 5. GENERAL AGRONOMIST ADVISORY MODE
    return f"""You are LeafLine Senior Agronomist and Crop Doctor.
{LEAFLINE_AGRONOMIST_SYSTEM_RULES}

FARMER QUESTION: {full_instruction}
CROP SUBJECT: {crop_name if crop_name else 'Agricultural Crop'}

CRITICAL INSTRUCTIONS:
1. Provide a helpful, clear, and structured answer directly addressing the farmer's question.
2. Use bullet points and bold headers for clarity and ease of reading in the field.
3. Ensure all recommendations are practical, realistic, and supported by the context.

CONTEXT FROM AGRICULTURAL GUIDES:
{context}

EXPERT AGRONOMIC RESPONSE:
"""


def extract_text_from_chunk(chunk) -> str:
    """Extract page content string safely from chunk or Document."""
    if hasattr(chunk, "page_content"):
        return chunk.page_content
    elif isinstance(chunk, dict):
        return chunk.get("text", "") or chunk.get("page_content", "")
    return str(chunk)