# -*- coding: utf-8 -*-
"""
app.py - Flask API Server for Crop Recommendation System
Endpoints:
  GET  /            → Serves the frontend
  GET  /model-info  → Returns model accuracies + feature importances
  POST /predict     → Returns predicted crop from 7 input features
  POST /explain     → Returns SHAP-based XAI explanation
"""

import os
import sys
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Path Setup ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR   = os.path.join(BASE_DIR, "static")

# Add backend to Python path so explainer.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explainer import CropExplainer

# ── Flask App ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)  # Allow cross-origin requests (for development)

# ── Load Artifacts ─────────────────────────────────────────────────────────
def load_artifacts():
    """Load trained models, preprocessors, and metadata."""
    models_exist = all(
        os.path.exists(os.path.join(MODEL_DIR, f))
        for f in ["rf_model.pkl", "lr_model.pkl", "nb_model.pkl",
                  "scaler.pkl", "label_encoder.pkl", "model_metadata.json"]
    )
    if not models_exist:
        print("\n[!] Models not found. Running train.py first...\n")
        import subprocess
        train_script = os.path.join(BASE_DIR, "backend", "train.py")
        subprocess.run([sys.executable, train_script], check=True)

    rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    lr_model = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
    nb_model = joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    le       = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)

    explainer = CropExplainer(rf_model, scaler, le, metadata["feature_names"])

    return rf_model, lr_model, nb_model, scaler, le, metadata, explainer


print("Loading models...")
rf_model, lr_model, nb_model, scaler, le, metadata, explainer = load_artifacts()
FEATURE_NAMES = metadata["feature_names"]
print("[OK] All models loaded. Starting server...")


# ── Helper ─────────────────────────────────────────────────────────────────
def parse_input(data: dict) -> list:
    """Parse and validate the 7 input features from request JSON."""
    keys = FEATURE_NAMES
    values = []
    for key in keys:
        val = data.get(key)
        if val is None:
            raise ValueError(f"Missing field: '{key}'")
        values.append(float(val))
    return values


def validate_ranges(values: list):
    """Validate input values are within reasonable agronomic ranges."""
    RANGES = {
        "N":           (0,   200,  "Nitrogen (N) should be 0–200 kg/ha"),
        "P":           (0,   150,  "Phosphorus (P) should be 0–150 kg/ha"),
        "K":           (0,   210,  "Potassium (K) should be 0–210 kg/ha"),
        "temperature": (0,   50,   "Temperature should be 0–50 °C"),
        "humidity":    (0,   100,  "Humidity should be 0–100 %"),
        "ph":          (0,   14,   "pH should be 0–14"),
        "rainfall":    (0,   3000, "Rainfall should be 0–3000 mm"),
    }
    for i, key in enumerate(FEATURE_NAMES):
        lo, hi, msg = RANGES[key]
        if not (lo <= values[i] <= hi):
            raise ValueError(msg)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend page."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return model accuracies, feature importances, and class names."""
    return jsonify({
        "status":   "ok",
        "metadata": metadata,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the most suitable crop.

    Request JSON: { "N": 90, "P": 42, "K": 43, "temperature": 20.9,
                    "humidity": 82.0, "ph": 6.5, "rainfall": 202.9 }

    Response JSON: { "crop": "rice", "emoji": "🌾",
                     "confidence": 95.2,
                     "all_proba": {...},
                     "model_predictions": {...} }
    """
    try:
        data   = request.get_json(force=True)
        values = parse_input(data)
        validate_ranges(values)

        X_input  = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        # Random Forest (primary)
        rf_pred_idx   = rf_model.predict(X_scaled)[0]
        rf_proba      = rf_model.predict_proba(X_scaled)[0]
        rf_crop       = le.inverse_transform([rf_pred_idx])[0]
        rf_confidence = round(float(rf_proba[rf_pred_idx]) * 100, 2)

        # Logistic Regression
        lr_pred_idx = lr_model.predict(X_scaled)[0]
        lr_crop     = le.inverse_transform([lr_pred_idx])[0]
        lr_proba    = lr_model.predict_proba(X_scaled)[0]
        lr_confidence = round(float(lr_proba[lr_pred_idx]) * 100, 2)

        # Naive Bayes
        nb_pred_idx = nb_model.predict(X_scaled)[0]
        nb_crop     = le.inverse_transform([nb_pred_idx])[0]
        nb_proba    = nb_model.predict_proba(X_scaled)[0]
        nb_confidence = round(float(nb_proba[nb_pred_idx]) * 100, 2)

        # Top-5 RF probabilities
        top5_idx  = np.argsort(rf_proba)[::-1][:5]
        top5      = {le.inverse_transform([i])[0]: round(float(rf_proba[i]) * 100, 2)
                     for i in top5_idx}

        from explainer import CROP_EMOJI
        emoji = CROP_EMOJI.get(rf_crop.lower(), "🌱")

        return jsonify({
            "status":     "ok",
            "crop":       rf_crop,
            "emoji":      emoji,
            "confidence": rf_confidence,
            "top5_crops": top5,
            "model_predictions": {
                "random_forest":       {"crop": rf_crop,  "confidence": rf_confidence},
                "logistic_regression": {"crop": lr_crop,  "confidence": lr_confidence},
                "naive_bayes":         {"crop": nb_crop,  "confidence": nb_confidence},
            },
        })

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """
    Return SHAP-based XAI explanation for the prediction.

    Same request body as /predict.
    """
    try:
        data   = request.get_json(force=True)
        values = parse_input(data)
        validate_ranges(values)

        result = explainer.explain(values)
        return jsonify({"status": "ok", **result})

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"XAI error: {str(e)}"}), 500


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Crop Recommendation API - Running")
    print("  http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
