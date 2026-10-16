import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Backend"))
sys.path.insert(0, str(ROOT_DIR / "Backend" / "Rag"))

from Backend.gemini_service import (
    is_gemini_configured,
    get_gemini_model_name,
)
from Backend.gemini_disease import analyze_plant_disease, MAX_IMAGE_SIZE_BYTES
from Backend.treatment_store import find_verified_treatment, load_treatments_data
from Backend.leafline_rag import get_rag_treatment_context
from Backend.classifier_bridge import (
    reconcile_predictions,
    predict_with_local_model,
    get_local_model_status,
    SUPPORTED_CROPS,
)
from Backend.Rag.pipeline import ask_leafline_agronomist
from Backend.Rag.retriever import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIR = ROOT_DIR / "frontend"
ASSETS_DIR = ROOT_DIR / "assets"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), template_folder=str(FRONTEND_DIR))
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB request ceiling
CORS(app)


# -------------------------------------------------------------
# Static HTML UI & Template Routes
# -------------------------------------------------------------
@app.route("/")
@app.route("/home")
@app.route("/home.html")
@app.route("/index.html")
def index():
    """Serves the Home landing page."""
    home_file = FRONTEND_DIR / "home.html"
    if home_file.exists():
        return send_from_directory(str(FRONTEND_DIR), "home.html")
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return send_from_directory(str(FRONTEND_DIR), "index.html")
    return send_from_directory(str(FRONTEND_DIR), "Chat.html")


@app.route("/chat")
@app.route("/chat.html")
@app.route("/Chat.html")
def chat():
    """Serves the LeafLine Disease Chat interface."""
    return send_from_directory(str(FRONTEND_DIR), "Chat.html")


@app.route("/roadmap")
@app.route("/roadmap.html")
@app.route("/RoadMap.html")
def roadmap():
    """Serves the LeafLine Crop Growth Roadmap interface."""
    return send_from_directory(str(FRONTEND_DIR), "RoadMap.html")


@app.route("/base")
@app.route("/base.html")
def base_template():
    """Serves the LeafLine Base Layout template."""
    return send_from_directory(str(FRONTEND_DIR), "base.html")


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    if ASSETS_DIR.exists():
        return send_from_directory(str(ASSETS_DIR), filename)
    return jsonify({"error": "Assets directory not found"}), 404


# -------------------------------------------------------------
# Health Check Endpoint
# -------------------------------------------------------------
@app.route("/health")
def health():
    try:
        store = get_vector_store()
        doc_count = store._collection.count()
    except Exception:
        doc_count = 0

    local_model_info = get_local_model_status()

    return jsonify({
        "status": "healthy",
        "service": "LeafLine Agritech AI",
        "gemini_configured": is_gemini_configured(),
        "gemini_model": get_gemini_model_name(),
        "local_model": local_model_info,
        "rag_indexed_chunks": doc_count,
        "supported_crops": list(SUPPORTED_CROPS.keys())
    }), 200


# -------------------------------------------------------------
# Multimodal Leaf & Crop Pathology Diagnosis Route
# -------------------------------------------------------------
@app.route("/analyze-disease", methods=["POST"])
@app.route("/predict", methods=["POST"])
@app.route("/api/v1/disease/analyze", methods=["POST"])
def analyze_disease():
    """
    Accepts an uploaded crop leaf photograph, performs vision analysis
    using Google Gemini Multimodal AI (via official google-genai SDK),
    identifies crop and condition, detects non-plant objects, quantifies uncertainty,
    reconciles with local classifier if present, and retrieves verified ICAR/TNAU treatments.
    """
    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({
            "success": False,
            "detail": "No image file provided. Please attach a crop leaf photograph.",
            "error_type": "missing_file"
        }), 400

    crop_hint = request.form.get("crop", "").strip() or None
    user_query = request.form.get("question", "").strip() or None

    try:
        image_bytes = file.read()

        # Enforce 5 MB upload limit
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            size_mb = len(image_bytes) / (1024 * 1024)
            return jsonify({
                "success": False,
                "detail": f"Image file exceeds the 5 MB limit (current size: {size_mb:.2f} MB). Please upload a smaller image.",
                "error_type": "file_too_large"
            }), 400

        # Step 1: Call Gemini Vision Analysis
        gemini_res = analyze_plant_disease(image_bytes, user_query=user_query)

        if not gemini_res.get("success"):
            err_type = gemini_res.get("error_type", "api_error")
            status_code = 400 if err_type == "validation_error" else (429 if err_type == "rate_limit" else 500)
            return jsonify({
                "success": False,
                "detail": gemini_res.get("message", "Plant image analysis failed."),
                "error_type": err_type
            }), status_code

        # Step 2: Check if a plant is visible
        if not gemini_res.get("is_plant_visible", False):
            return jsonify({
                "success": False,
                "is_plant_visible": False,
                "crop": "None",
                "disease_name": "Non-Plant Subject",
                "predicted_disease": "Non-Plant Subject",
                "visible_symptoms": [],
                "assessment": gemini_res.get("assessment", "No plant or agricultural specimen is visible in the provided image."),
                "uncertainty": gemini_res.get("uncertainty", "High"),
                "reasoning": gemini_res.get("reasoning", "The uploaded photograph does not contain plant foliage."),
                "detail": "No plant leaf was detected in the photograph. Please upload a clear photo of an agricultural plant leaf."
            }), 400

        # Step 3: Reconcile with local model if available
        local_pred = predict_with_local_model(image_bytes)
        consensus = reconcile_predictions(gemini_res, local_pred)

        crop = consensus["final_crop"] or crop_hint or "Agricultural Plant"
        disease_name = consensus["final_disease"]

        # Step 4: Retrieve verified treatments from verified knowledge base
        treatment_record = find_verified_treatment(crop, disease_name)

        # Step 5: Optional RAG context enrichment from ICAR/TNAU ChromaDB
        rag_context = get_rag_treatment_context(crop, disease_name)

        # Step 6: Construct structured, safe response
        response_data = {
            "success": True,
            "is_plant_visible": True,
            "crop": crop.lower().split()[0] if crop else "plant",
            "crop_name": crop,
            "disease_name": disease_name,
            "predicted_disease": disease_name,
            "visible_symptoms": gemini_res.get("visible_symptoms", []),
            "assessment": gemini_res.get("assessment", ""),
            "uncertainty": consensus.get("uncertainty_status", gemini_res.get("uncertainty", "")),
            "reasoning": gemini_res.get("reasoning", ""),
            "is_crop_supported": consensus.get("is_crop_supported", True),
            "crop_support_note": consensus.get("crop_support_note", ""),
            "disagreement": consensus.get("disagreement", False),
            "verified_treatment": treatment_record,
            "rag_context": rag_context,
            "status": "confirmed" if treatment_record.get("is_verified") else "preliminary_assessment",
            "confidence": 0.95 if treatment_record.get("is_verified") else None,
            "treatment_guidance": {
                "overview": treatment_record.get("overview") or gemini_res.get("assessment"),
                "symptoms": treatment_record.get("symptoms") or gemini_res.get("visible_symptoms"),
                "non_chemical_control": treatment_record.get("non_chemical_management", []),
                "verified_treatment": treatment_record.get("treatment", []),
                "prevention": treatment_record.get("prevention", []),
                "safety_precautions": treatment_record.get("safety_precautions", [])
            },
            "source_references": treatment_record.get("source_references", []),
            "model": gemini_res.get("model", "Gemini Vision")
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "detail": "An internal error occurred during plant disease analysis. Please try again.",
            "error_type": "server_error"
        }), 500


# -------------------------------------------------------------
# Direct Verified Treatment Lookup Route
# -------------------------------------------------------------
@app.route("/treatment-info/<path:raw_label>", methods=["GET"])
def treatment_info_endpoint(raw_label: str):
    """
    Direct lookup endpoint for verified treatment guidance by raw label or disease name.
    """
    clean_label = raw_label.strip() if raw_label else ""
    if not clean_label:
        return jsonify({"error": "Missing raw_label parameter."}), 400

    parts = clean_label.split("___", 1)
    crop = parts[0].replace("_", " ") if len(parts) > 0 else clean_label
    disease = parts[1].replace("_", " ") if len(parts) > 1 else clean_label

    treatment = find_verified_treatment(crop, disease)
    return jsonify(treatment), 200


# -------------------------------------------------------------
# LeafLine Agricultural RAG Knowledge Base Route
# -------------------------------------------------------------
@app.route("/api/v1/rag/ask", methods=["POST"])
@app.route("/api/rag/ask", methods=["POST"])
def rag_ask():
    """
    Answers farmer questions using the verified agricultural PDF guidebooks
    indexed in ChromaDB (Beans, Brinjal, Chilli, Cucurbits, Okra, Spinach, Tomato, etc.).
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"detail": "Please provide a question."}), 400

    crop = data.get("crop", "").strip() or None
    history = data.get("history", [])

    try:
        result = ask_leafline_agronomist(
            question=question,
            history=history,
            crop_filter=crop
        )
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        return jsonify({"detail": f"RAG query failed: {str(e)}"}), 500


# -------------------------------------------------------------
# Crop Growth Roadmap Routes
# -------------------------------------------------------------
ROADMAPPED_CROPS = [
    {
        "crop_id": "tomato",
        "common_names": ["Tomato"],
        "scientific_name": "Solanum lycopersicum",
        "stage_count": 5,
        "evidence_status": "verified"
    },
    {
        "crop_id": "chilli",
        "common_names": ["Chilli", "Hot Pepper"],
        "scientific_name": "Capsicum annuum",
        "stage_count": 4,
        "evidence_status": "verified"
    },
    {
        "crop_id": "brinjal",
        "common_names": ["Brinjal", "Eggplant"],
        "scientific_name": "Solanum melongena",
        "stage_count": 4,
        "evidence_status": "verified"
    },
    {
        "crop_id": "beans",
        "common_names": ["French Beans"],
        "scientific_name": "Phaseolus vulgaris",
        "stage_count": 3,
        "evidence_status": "verified"
    },
    {
        "crop_id": "cucumber",
        "common_names": ["Cucumber"],
        "scientific_name": "Cucumis sativus",
        "stage_count": 4,
        "evidence_status": "verified"
    },
    {
        "crop_id": "tree_planting",
        "common_names": ["Tree Planting Guide", "Trees & Saplings"],
        "scientific_name": "Universal Sapling & Fruit Tree Care",
        "stage_count": 8,
        "evidence_status": "verified"
    }
]

CROP_ROADMAP_DETAILS = {
    "tomato": {
        "crop_id": "tomato",
        "common_names": ["Tomato"],
        "scientific_name": "Solanum lycopersicum L.",
        "evidence_status": "verified",
        "identity_note": "Solanum lycopersicum (Tomato) — 16-Week Day-by-Day Lifecycle & Agronomic Care.",
        "supported_methods": ["transplanting", "raised_beds", "staking", "drip_fertigation"],
        "applicability_note": "TNAU Agritech and ICAR IIHR package of practices for tomato crop health.",
        "focus_stages": ["nursery", "transplanting", "vegetative", "flowering", "harvesting"],
        "stages": [
            {
                "stage_id": "nursery",
                "stage_name": "Nursery, Sowing & Baby Sprouts",
                "timing_display": "Days 1–25 (After Sowing)",
                "timing_basis": "days_after_sowing",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Sow seeds 0.5 cm deep in coco peat + vermicompost trays",
                    "Dust seeds with Trichoderma viride (4g/kg) to stop damping-off root rot",
                    "Give 6–8 hours bright morning sun the moment green shoots peek through",
                    "Water soil base gently; allow top layer to dry slightly between waterings"
                ],
                "pest_disease_watch": ["Damping-off root rot (Pythium fungi)", "Flea beetles & fungus gnats"],
                "nutrient_advice": ["Avoid chemical fertilizers; starter vermicompost supplies all seedling food"],
                "irrigation_advice": ["Gentle morning misting; keep soil like a squeezed sponge (never waterlogged)"]
            },
            {
                "stage_id": "transplanting",
                "stage_name": "Hardening Off & Evening Transplanting",
                "timing_display": "Days 25–40 (25-Day Seedlings)",
                "timing_basis": "days_after_transplanting",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Harden off outdoors in dappled shade for 3–4 days before moving to beds/pots",
                    "Transplant in cool late afternoon (4–6 PM) with 60x45 cm spacing",
                    "Bury stems deep right up to the first leaves so extra anchor roots sprout",
                    "Spread 5 cm organic straw or leaf mulch around base to keep roots cool"
                ],
                "pest_disease_watch": ["Cutworms at soil level (use paper collars)", "Early blight leaf spots"],
                "nutrient_advice": ["Mix 30% vermicompost + 50g neem cake (kills soil root-knot nematodes naturally)"],
                "irrigation_advice": ["Water deeply immediately after transplanting; water every 2–3 days at base only"]
            },
            {
                "stage_id": "vegetative",
                "stage_name": "Staking, Trellising & Sucker Pruning",
                "timing_display": "Days 40–55 (Rapid Vine Growth)",
                "timing_basis": "days_after_transplanting",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Tie main stem to 5-foot bamboo stakes using loose figure-8 cloth or twine ties",
                    "Pinch off side-shoots ('suckers' under 5 cm) growing in branch armpits",
                    "Prune off yellowing lower leaves within 20 cm of soil for bottom airflow",
                    "Top-dress with 1 handful vermicompost and 20g wood ash around drip line"
                ],
                "pest_disease_watch": ["Early Blight (Alternaria fungi)", "Whiteflies (vectors of Leaf Curl Virus)"],
                "nutrient_advice": ["Balanced organic compost + potassium (wood ash); avoid excessive chemical nitrogen"],
                "irrigation_advice": ["Deep morning base watering; never splash water onto leaves"]
            },
            {
                "stage_id": "flowering",
                "stage_name": "Yellow Blossoms, Pollination & Fruit Set",
                "timing_display": "Days 55–75 (Flowering to Fruit Swell)",
                "timing_basis": "days_after_transplanting",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Gently tap or vibrate bamboo stakes at midday to help self-pollination",
                    "Maintain steady, uniform soil moisture to prevent flower drop & Blossom-End Rot",
                    "Hang yellow sticky traps 30 cm above canopy for whitefly & thrips control",
                    "Support heavy fruit clusters with soft twine ties to prevent stem kinking"
                ],
                "pest_disease_watch": ["Tomato Fruit Borer (Helicoverpa armigera)", "Blossom-End Rot (black sunken fruit bottoms)"],
                "nutrient_advice": ["Apply 2 handfuls compost + bone meal; spray 0.5% Calcium Nitrate if blossom-end rot appears"],
                "irrigation_advice": ["Keep soil evenly moist like a wrung-out sponge; avoid swings between drought and flooding"]
            },
            {
                "stage_id": "harvesting",
                "stage_name": "Breaker Stage Ripening & Harvest",
                "timing_display": "Days 75–110+ (Harvest Window)",
                "timing_basis": "days_after_transplanting",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Harvest at Breaker Stage (first blush of yellow/pink) for maximum flavor and zero bird damage",
                    "Pick in cool morning hours; snap stem at natural joint or clip with clean shears",
                    "Never refrigerate tomatoes — keep at room temperature (above 15°C / 60°F) to preserve sweet aroma",
                    "Top indeterminate vines by pinching top tip 3 weeks before season end to size remaining fruit"
                ],
                "pest_disease_watch": ["Fruit flies", "Late Blight (Phytophthora)", "Bird pecks on red fruit"],
                "nutrient_advice": ["Scratch 1 handful vermicompost into topsoil every 3 weeks to support ongoing upper clusters"],
                "irrigation_advice": ["Slightly moderate watering during harvest to concentrate sugars and stop fruit splitting"]
            }
        ]
    },
    "chilli": {
        "crop_id": "chilli",
        "common_names": ["Chilli", "Hot Pepper"],
        "scientific_name": "Capsicum annuum L.",
        "evidence_status": "verified",
        "identity_note": "Capsicum annuum (Chilli) — 16-Week Day-by-Day Lifecycle & Agronomic Care.",
        "supported_methods": ["transplanting", "raised_beds", "ipm_vector_control"],
        "applicability_note": "TNAU Agritech and ICAR IIHR package of practices for chilli crop health.",
        "focus_stages": ["nursery", "transplanting", "flowering", "harvesting"],
        "stages": [
            {
                "stage_id": "nursery",
                "stage_name": "Nursery & Emergence",
                "timing_display": "Days 1–30",
                "timing_basis": "days_after_sowing",
                "evidence_status": "verified",
                "critical_tasks": ["Seed treatment with Thiram or Trichoderma", "Cover nursery with 40-mesh insect net"],
                "pest_disease_watch": ["Damping off", "Thrips"]
            },
            {
                "stage_id": "transplanting",
                "stage_name": "Transplanting & Rooting",
                "timing_display": "Days 30–45",
                "timing_basis": "days_after_transplanting",
                "evidence_status": "verified",
                "critical_tasks": ["Plant 30-day seedlings at 45x30cm spacing", "Irrigate immediately"],
                "pest_disease_watch": ["Cutworms"]
            },
            {
                "stage_id": "flowering",
                "stage_name": "Flowering & Pod Set",
                "timing_display": "Days 60–85",
                "timing_basis": "days_after_transplanting",
                "evidence_status": "verified",
                "critical_tasks": ["Install yellow sticky traps (12/acre)", "Spray neem oil 3ml/L for sucking pests"],
                "pest_disease_watch": ["Chilli Leaf Curl (Gemini virus)", "Thrips", "Mites"]
            },
            {
                "stage_id": "harvesting",
                "stage_name": "Green & Red Pod Harvest",
                "timing_display": "Days 85–120",
                "timing_basis": "days_after_transplanting",
                "evidence_status": "verified",
                "critical_tasks": ["Periodic picking of green or ripe red pods", "Dry on clean tarpaulin"],
                "pest_disease_watch": ["Anthracnose / Die-back (Colletotrichum)"]
            }
        ]
    },
    "brinjal": {
        "crop_id": "brinjal",
        "common_names": ["Brinjal", "Eggplant"],
        "scientific_name": "Solanum melongena L.",
        "evidence_status": "verified",
        "identity_note": "Solanum melongena (Brinjal) — 16-Week Day-by-Day Lifecycle & Agronomic Care.",
        "supported_methods": ["transplanting", "ipm_borer_management"],
        "applicability_note": "TNAU Agritech and ICAR package of practices.",
        "focus_stages": ["nursery", "transplanting", "flowering", "harvesting"],
        "stages": [
            {
                "stage_id": "nursery",
                "stage_name": "Nursery Stage",
                "timing_display": "Days 1–30",
                "timing_basis": "days_after_sowing",
                "evidence_status": "verified",
                "critical_tasks": ["Seed treatment", "Raised bed care"],
                "pest_disease_watch": ["Damping off"]
            },
            {
                "stage_id": "transplanting",
                "stage_name": "Field Transplanting",
                "timing_display": "Days 30–45",
                "timing_basis": "days_after_transplanting",
                "evidence_status": "verified",
                "critical_tasks": ["Spacing 60x60cm", "Basal manure application"],
                "pest_disease_watch": ["Bacterial Wilt"]
            },
            {
                "stage_id": "flowering",
                "stage_name": "Branching & Fruit Setting",
                "timing_display": "Days 45–90",
                "timing_basis": "days_after_transplanting",
                "evidence_status": "verified",
                "critical_tasks": ["Install pheromone traps for shoot & fruit borer (10/ha)", "Clip wilted shoots regularly"],
                "pest_disease_watch": ["Shoot & Fruit Borer (Leucinodes orbonalis)", "Little leaf"]
            },
            {
                "stage_id": "harvesting",
                "stage_name": "Harvesting",
                "timing_display": "Days 90–120",
                "timing_basis": "days_after_transplanting",
                "evidence_status": "verified",
                "critical_tasks": ["Harvest tender glossy fruits", "Pick every 4–5 days"],
                "pest_disease_watch": ["Phomopsis blight"]
            }
        ]
    },
    "tree_planting": {
        "crop_id": "tree_planting",
        "common_names": ["Tree Planting Guide", "Trees & Saplings"],
        "scientific_name": "Universal Sapling & Fruit Tree Care",
        "evidence_status": "verified",
        "identity_note": "Universal 8-Stage Tree & Sapling Planting Master Guide — Grounded in ICAR-CAZRI, Forest Research Institute (FRI) & FAO Guidelines.",
        "supported_methods": ["pit_solarization", "organic_neem_shield", "saucer_basin_irrigation", "3-3-3_mulching", "flexible_staking"],
        "applicability_note": "Grounded in ICAR-CAZRI and FAO agroforestry manuals; applicable to fruit trees (mango, guava, citrus, sapota, pomegranate) and shade/forest trees across all regions.",
        "focus_stages": ["siting", "pit_prep", "soil_enrichment", "planting", "irrigation", "mulching", "staking", "year1_care"],
        "stages": [
            {
                "stage_id": "siting",
                "stage_name": "Site Selection, Sunlight & Distance Planning",
                "timing_display": "3–4 Weeks Before Planting",
                "timing_basis": "planning_phase",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Pick a planting spot that receives at least 6 to 8 hours of direct sunshine daily",
                    "Keep big shade trees 15 to 20 feet (5–6 meters) away from house walls, septic tanks, and paved driveways",
                    "Keep fruit trees at least 10 to 12 feet apart for healthy air circulation and fruit ripening",
                    "Test drainage: Dig a 1-foot test hole, fill with water. If it drains within 3 hours, drainage is ideal!"
                ],
                "pest_disease_watch": ["Waterlogged standing puddle water", "Overhead electrical power lines", "Heavy shade from high walls"],
                "nutrient_advice": ["No fertilizer needed during siting; focus on finding clean, deep, well-draining soil"],
                "irrigation_advice": ["Verify that your planting location has easy access to a clean water source for weekly deep watering"]
            },
            {
                "stage_id": "pit_prep",
                "stage_name": "Digging the 3×3×3 Pit & Sun Solarization",
                "timing_display": "2–3 Weeks Before Planting",
                "timing_basis": "pit_preparation",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Dig a square pit measuring 1 meter wide, 1 meter long, and 1 meter deep (3 ft × 3 ft × 3 ft)",
                    "Keep dark, fertile topsoil in one pile, and lower reddish subsoil in a separate pile",
                    "Leave the open pit baking in bright sunshine for 10 to 15 days ('Solarization')",
                    "Break up hard rock or clay at the pit floor with a crowbar so taproots can dive deep into the earth"
                ],
                "pest_disease_watch": ["Hidden termite colonies in surrounding soil", "Hard rocky impenetrable subsoil layer"],
                "nutrient_advice": ["Sun solarization naturally destroys weed roots, nematode eggs, and soil fungi without harmful chemicals"],
                "irrigation_advice": ["Keep the open pit completely dry during solarization so intense sun heat purifies the soil"]
            },
            {
                "stage_id": "soil_enrichment",
                "stage_name": "Pit Soil Mixture & Organic Termite Shield",
                "timing_display": "3–5 Days Before Planting",
                "timing_basis": "soil_enrichment",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Mix topsoil with 15–20 kg well-rotted farmyard manure (cow compost) or dark vermicompost",
                    "Add 500 grams of pure Neem Cake powder (vital organic shield to repel underground termites & root grubs)",
                    "Add 50 grams Trichoderma viride bio-fungicide to safeguard against root-rot fungi",
                    "Add 100 grams bone meal or rock phosphate (slow-release natural phosphorus for fast root branching)",
                    "Refill pit with enriched mixture and water lightly so loose soil settles naturally before planting"
                ],
                "pest_disease_watch": ["Subterranean termites", "White grubs", "Root-knot nematodes"],
                "nutrient_advice": ["Never use fresh, wet, smelly cow dung (it heats up and burns roots); use only cool, aged, earthy compost"],
                "irrigation_advice": ["Pre-soak refilled soil 3 days before planting so soil settles to natural ground level"]
            },
            {
                "stage_id": "planting",
                "stage_name": "Zero-Mistake Sapling Planting Day",
                "timing_display": "Day 0 (Planting Day - Late Afternoon 4–6 PM)",
                "timing_basis": "planting_day",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Carefully slice and remove the black nursery polybag; NEVER yank the stem or shatter the soil ball",
                    "Gently untangle any winding roots at the bottom with your fingers so roots grow outward into the garden",
                    "CRUCIAL RULE: Position the root flare (or graft joint) 2 inches ABOVE the soil surface — NEVER bury the green trunk!",
                    "Backfill enriched soil in layers and press down gently with your feet to eliminate underground air pockets",
                    "Plant in the cool late afternoon (4–6 PM) to give young roots 12 hours of cool recovery overnight"
                ],
                "pest_disease_watch": ["Collar rot (bark rotting from being planted too deep — #1 killer of young trees worldwide!)", "Air pocket root drying"],
                "nutrient_advice": ["Do not add chemical urea or synthetic fertilizers to planting hole; organic compost provides all initial nutrition"],
                "irrigation_advice": ["Water the nursery bag thoroughly the evening before planting so the soil ball stays firm and intact"]
            },
            {
                "stage_id": "irrigation",
                "stage_name": "The Saucer Basin & First 20-Liter Deep Drench",
                "timing_display": "Immediately on Planting Day",
                "timing_basis": "post_planting",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Build a raised circular soil ring (bund) 10 cm (4 inches) high and 1 meter wide around the sapling",
                    "This saucer basin catches and holds water directly over the root ball without splashing against the bark",
                    "Give the first deep drench: slowly pour 15 to 20 liters (approx. 2 full buckets) into the saucer basin",
                    "Check for sinkholes: if soil sinks near the trunk after watering, top up with 1 trowel of dry topsoil immediately"
                ],
                "pest_disease_watch": ["Soil sinkholes causing trunk tilting", "Water splashing directly against green bark"],
                "nutrient_advice": ["Water dissolved organic compost tea into the root zone to promote instant microbial activity"],
                "irrigation_advice": ["Water inside the saucer ring, never directly against the trunk bark. Water slowly until fully absorbed"]
            },
            {
                "stage_id": "mulching",
                "stage_name": "The 3-3-3 Mulching Blanket (Natural Soil Shield)",
                "timing_display": "Day 1 After Planting",
                "timing_basis": "mulching_phase",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Spread an organic mulch blanket of dry leaves, rice straw, sugarcane bagasse, or wood chips",
                    "Follow the famous 3-3-3 Rule: 3 inches thick, in a 3-foot wide ring, keeping 3 inches of clear space away from the trunk",
                    "NEVER pile mulch against the trunk like a 'mulch volcano' — wet mulch against bark causes fungal rot and beetle attack",
                    "Mulch keeps root soil 5°C cooler in blistering heat, blocks weeds, and cuts evaporation water loss by 60%"
                ],
                "pest_disease_watch": ["Mulch volcano rot on trunk", "Weed competition in root basin"],
                "nutrient_advice": ["Organic mulch slowly breaks down over 6 months into rich humus, continuously nourishing earthworms and roots"],
                "irrigation_advice": ["Water passes right through dry mulch into the soil below while the surface stays clean and weed-free"]
            },
            {
                "stage_id": "staking",
                "stage_name": "Bamboo Staking & Soft Figure-8 Wind Ties",
                "timing_display": "Days 1–7 After Planting",
                "timing_basis": "staking_phase",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Drive a sturdy 6-foot bamboo or wooden pole 40 cm deep into firm ground, 20 cm outside the root ball",
                    "Tie the trunk to the stake using a soft cloth strip or bicycle rubber tube in a loose figure-8 knot",
                    "Allow 2 to 3 inches of natural sway in breezes — swaying signals the tree to grow a thicker, storm-resistant trunk!",
                    "NEVER use thin wire or tight nylon cords that strangle and cut into growing bark as the tree expands"
                ],
                "pest_disease_watch": ["Bark abrasion from rubbing against stake", "Stem girdling from tight string ties"],
                "nutrient_advice": ["Healthy mechanical flexing stimulates internal auxin hormone production for dense wood growth"],
                "irrigation_advice": ["Ensure the stake is driven outside the root ball so feeder roots are never punctured or damaged"]
            },
            {
                "stage_id": "year1_care",
                "stage_name": "Year 1 Seasonal Care, Tree Guard & Protection",
                "timing_display": "Months 1–12 (Full First Year Journey)",
                "timing_basis": "first_year_management",
                "evidence_status": "verified",
                "critical_tasks": [
                    "Install a 4-to-5 foot wire mesh tree guard to protect tender bark from browsing goats, cattle, and dogs",
                    "Watering routine: Deep soak with 15–20 liters 1 to 2 times a week during dry months (never shallow daily sprinkles)",
                    "Fruit trees: Snip off any wild 'sucker' shoots growing below the graft union on the lower trunk immediately",
                    "Summer defense: Paint lower 45 cm of trunk with white slaked lime (chuna) to reflect blistering sunrays",
                    "Feed at Month 3 and Month 9 with 5 kg vermicompost and 100g wood ash scratched lightly under the mulch"
                ],
                "pest_disease_watch": ["Grazing animals stripping bark", "Citrus leaf miners (silvery trails on leaves)", "Wild rootstock suckers stealing food"],
                "nutrient_advice": ["Top-dress with vermicompost every 3 months around the drip line (outer edge of branch canopy)"],
                "irrigation_advice": ["Check soil 2 inches below mulch: if damp and cool, skip watering; if dry, give a deep 20-liter basin soak"]
            }
        ]
    }
}


@app.route("/api/v1/crops", methods=["GET"])
def get_crops():
    return jsonify(ROADMAPPED_CROPS), 200


@app.route("/api/v1/crops/<crop_id>/roadmap", methods=["GET"])
def get_crop_roadmap(crop_id):
    crop_key = crop_id.lower().strip()
    if crop_key in CROP_ROADMAP_DETAILS:
        return jsonify(CROP_ROADMAP_DETAILS[crop_key]), 200
    stub = {
        "crop_id": crop_key,
        "common_names": [crop_key.capitalize()],
        "scientific_name": f"{crop_key.capitalize()} sp.",
        "evidence_status": "partially_verified",
        "identity_note": f"Crop Growth Roadmap for {crop_key.capitalize()}.",
        "supported_methods": ["cultivation", "ipm"],
        "applicability_note": "Agritech package of practices.",
        "focus_stages": ["germination", "vegetative", "harvest"],
        "stages": [
            {
                "stage_id": "germination",
                "stage_name": "Sowing & Germination",
                "timing_display": "Days 1–15",
                "timing_basis": "days_after_sowing",
                "evidence_status": "verified",
                "critical_tasks": ["Seed treatment and soil preparation"],
                "pest_disease_watch": ["Damping off"]
            },
            {
                "stage_id": "harvest",
                "stage_name": "Maturity & Harvesting",
                "timing_display": "Days 60–90",
                "timing_basis": "days_after_sowing",
                "evidence_status": "verified",
                "critical_tasks": ["Timely harvesting and handling"],
                "pest_disease_watch": ["Fruit rot"]
            }
        ]
    }
    return jsonify(stub), 200


# -------------------------------------------------------------
# Dual WSGI & ASGI Compatibility Wrapper
# Allows running with either:
#   - Standard Python: python main.py
#   - ASGI / Uvicorn:  uvicorn main:app --reload --port 8000
# -------------------------------------------------------------
try:
    from a2wsgi import WSGIMiddleware

    class _HybridASGIApp:
        """Adapts Flask WSGI app to support ASGI servers like Uvicorn seamlessly."""
        def __init__(self, inner_app):
            self._inner = inner_app
            self._asgi = WSGIMiddleware(inner_app.wsgi_app)

        async def __call__(self, scope, receive, send):
            return await self._asgi(scope, receive, send)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    asgi_app = _HybridASGIApp(app)
    app = asgi_app
except ImportError:
    pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🌱 LeafLine Agritech AI running at: http://127.0.0.1:{port}/")
    print(f"💬 Disease Chat: http://127.0.0.1:{port}/chat\n")
    app.run(host="0.0.0.0", port=port, debug=False)

