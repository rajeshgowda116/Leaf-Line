import io
import os
import sys
import webbrowser
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image

# Ensure PyTorch backend for Keras 3 if TensorFlow is not installed (e.g. Python 3.13)
os.environ.setdefault("KERAS_BACKEND", "torch")
try:
    from tensorflow import keras
except ImportError:
    import keras

# Import treatment guidance service and leaf validation
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from treatment_service import get_treatment_info
from leaf_validator import validate_image_file, is_plant_leaf

# Resolve model path reliably whether invoked from repo root or backend/
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
if not MODEL_PATH.exists():
    MODEL_PATH = CURRENT_DIR / "models" / "plant_disease_model.keras"

# Disable torch dynamo compiler if running with torch backend (prevents missing MSVC compiler error on Windows)
try:
    import torch
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True
except Exception:
    pass

model = keras.models.load_model(str(MODEL_PATH))
try:
    model.compile(jit_compile=False)
except Exception:
    pass

# Class names (38 classes recognized by LeafLine model in standard alphabetical order)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Verify model output dimensionality matches classes
if hasattr(model, "output_shape") and model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f"Model output dimension ({model.output_shape[-1]}) does not match "
        f"CLASS_NAMES length ({len(CLASS_NAMES)})"
    )

import datetime

IMG_SIZE = (128, 128)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

_PREDICTION_HISTORY = []

app = Flask(__name__, static_folder=FRONTEND_DIR)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_BYTES
CORS(app)


def preprocess_image_from_pil(image: Image.Image) -> np.ndarray:
    """Preprocesses a PIL image for model inference matching training parameters."""
    resized = image.resize(IMG_SIZE)
    img_array = np.array(resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocesses raw bytes into model input tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return preprocess_image_from_pil(image)


def format_class_name(raw_class: str) -> str:
    """Replaces triple underscores and double underscores with comma and space."""
    return raw_class.replace('___', ', ').replace('__', ', ').replace('_', ' ')


def parse_plant_and_status(class_label: str):
    """Splits raw model label into clean plant name and disease status."""
    if '___' in class_label:
        plant, status = class_label.split('___', 1)
        plant = plant.replace('_', ' ').replace(',', ', ')
        status = status.replace('_', ' ').replace(',', ', ')
    else:
        plant = class_label.replace('_', ' ').replace(',', ', ')
        status = "Unknown"
    return plant.strip().title(), status.strip().title()


# -------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "is_leaf": False,
        "error": "File size exceeds the maximum allowed limit of 10.0 MB. Please upload a smaller image."
    }), 413


@app.errorhandler(400)
def bad_request_handler(error):
    desc = getattr(error, 'description', "Bad request.")
    return jsonify({
        "success": False,
        "is_leaf": False,
        "error": desc
    }), 400


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "success": False,
        "error": "An internal server error occurred while processing the request."
    }), 500


# -------------------------------------------------------------
# Prediction Route
# -------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "is_leaf": False,
            "error": "No file part in request."
        }), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({
            "success": False,
            "is_leaf": False,
            "error": "No file selected. Please select a leaf image to upload."
        }), 400

    try:
        img_bytes = file.read()

        # Stage 1: File integrity and image decoding validation
        is_valid_file, file_msg, pil_img = validate_image_file(img_bytes, max_size_bytes=MAX_FILE_SIZE_BYTES)
        if not is_valid_file or pil_img is None:
            return jsonify({
                "success": False,
                "is_leaf": False,
                "error": file_msg
            }), 400

        # Stage 2: Plant-Leaf Presence Validation
        is_leaf, leaf_conf, leaf_msg, leaf_meta = is_plant_leaf(pil_img)
        if not is_leaf:
            try:
                _PREDICTION_HISTORY.append({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": file.filename,
                    "success": False,
                    "is_leaf": False,
                    "plant": "Non-Plant Object",
                    "status": "Validation Rejected",
                    "confidence": "N/A",
                    "raw_label": "not_plant_leaf",
                    "is_inconclusive": True,
                    "message": leaf_msg
                })
            except Exception:
                pass
            return jsonify({
                "success": False,
                "is_leaf": False,
                "error": leaf_msg,
                "details": leaf_meta
            }), 400

        # Stage 3: Plant Disease Model Inference
        img_array = preprocess_image_from_pil(pil_img)
        preds = model.predict(img_array, verbose=0)[0]

        # Stage 4: Output Validation
        if len(preds) != len(CLASS_NAMES) or np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            return jsonify({
                "success": False,
                "error": "Model produced invalid or non-finite output dimensions."
            }), 500

        # Stage 5: Confidence & Uncertainty Quantification
        preds_float = np.array(preds, dtype=np.float64)
        # Ensure positive probabilities summing to 1
        preds_float = np.clip(preds_float, 0.0, 1.0)
        p_sum = np.sum(preds_float)
        if p_sum > 0:
            preds_float = preds_float / p_sum

        sorted_indices = np.argsort(preds_float)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1])

        p1 = float(preds_float[top1_idx])
        p2 = float(preds_float[top2_idx])
        margin = float(p1 - p2)

        # Normalized Shannon Entropy H_hat in [0, 1]
        eps = 1e-12
        entropy_raw = -float(np.sum(preds_float * np.log(preds_float + eps)))
        entropy_max = float(np.log(len(CLASS_NAMES)))
        normalized_entropy = float(entropy_raw / entropy_max) if entropy_max > 0 else 0.0

        # Calibrated Uncertainty Policy:
        # Inconclusive if top confidence < 45%, margin < 15%, or entropy > 0.65
        is_inconclusive = bool(p1 < 0.45 or margin < 0.15 or normalized_entropy > 0.65)

        raw_label = CLASS_NAMES[top1_idx]
        plant, status = parse_plant_and_status(raw_label)
        confidence_percent = p1 * 100.0

        if is_inconclusive:
            status_display = "Inconclusive"
            advisory_message = (
                "Prediction is inconclusive due to diagnostic uncertainty or ambiguous visual patterns. "
                "Agrochemicals must not be applied without in-person confirmation by a certified agronomist or KVK officer."
            )
        else:
            status_display = status
            advisory_message = (
                "Visual prediction generated. Always verify with an agricultural expert before taking chemical action."
            )

        # Audit Logging
        log_file_path = os.path.join(os.path.dirname(__file__), "prediction_log.txt")
        try:
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"File: {file.filename}, Plant: {plant}, Status: {status_display}, "
                    f"Tentative: {status}, Confidence: {confidence_percent:.2f}%, Margin: {margin:.4f}, "
                    f"Entropy: {normalized_entropy:.4f}, Inconclusive: {is_inconclusive}, "
                    f"Raw: {raw_label}\n"
                )
        except Exception:
            pass

        # Record in in-memory history
        try:
            _PREDICTION_HISTORY.append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": file.filename,
                "success": True,
                "is_leaf": True,
                "plant": plant,
                "status": status_display,
                "tentative_condition": status,
                "raw_label": raw_label,
                "confidence": f"{confidence_percent:.2f}%",
                "is_inconclusive": is_inconclusive,
                "message": advisory_message
            })
        except Exception:
            pass

        return jsonify({
            "success": True,
            "plant": plant,
            "status": status_display,
            "tentative_condition": status,
            "raw_label": raw_label,
            "confidence": f"{confidence_percent:.2f}%",
            "confidence_score": round(p1, 4),
            "top2_confidence_score": round(p2, 4),
            "margin": round(margin, 4),
            "entropy": round(normalized_entropy, 4),
            "is_inconclusive": is_inconclusive,
            "is_leaf": True,
            "message": advisory_message
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -------------------------------------------------------------
# Prediction History Endpoints
# -------------------------------------------------------------
@app.route("/api/history", methods=["GET"])
def get_prediction_history():
    """Returns the most recent 30 prediction logs (newest first)."""
    return jsonify({"history": _PREDICTION_HISTORY[-30:][::-1]})


@app.route("/api/history", methods=["DELETE"])
def clear_prediction_history():
    """Clears the session prediction history."""
    _PREDICTION_HISTORY.clear()
    return jsonify({"success": True, "message": "Prediction history cleared."})


# -------------------------------------------------------------
# Treatment Guidance & Marketplace Endpoint
# -------------------------------------------------------------
@app.route("/treatment-info/<path:raw_label>", methods=["GET"])
def treatment_info_endpoint(raw_label):
    """
    Returns disease management guidance, prevention steps,
    and external verified marketplace search links for the given raw_label.
    Supports ?inconclusive=true query parameter to suppress chemical advice.
    """
    clean_label = raw_label.strip() if raw_label else ""
    if not clean_label:
        return jsonify({
            "error": "Missing raw_label parameter.",
            "status_type": "invalid"
        }), 400

    is_inconclusive = request.args.get("inconclusive", "false").lower() in ("true", "1", "yes")

    info = get_treatment_info(clean_label, is_inconclusive=is_inconclusive)
    if info.get("status_type") == "unrecognized":
        return jsonify(info), 404
    elif info.get("status_type") == "invalid":
        return jsonify(info), 400

    return jsonify(info), 200


@app.route("/treatment-info/", methods=["GET"])
def treatment_info_empty():
    return jsonify({
        "error": "Missing raw_label parameter.",
        "status_type": "invalid"
    }), 400


# -------------------------------------------------------------
# Static Frontend Serving
# -------------------------------------------------------------
@app.route("/")
def root():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/about")
def about():
    return send_from_directory(FRONTEND_DIR, "about.html")


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true")

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8000/")

    threading.Timer(1.5, open_browser).start()
    app.run(host="0.0.0.0", port=8000, debug=debug_mode)
