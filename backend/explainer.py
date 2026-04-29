"""
explainer.py — XAI Explanation Engine
Uses SHAP TreeExplainer to explain Random Forest predictions.
Converts SHAP values into human-readable text explanations.
"""

import shap
import numpy as np
import joblib
import os

# ── Feature display names and units ───────────────────────────────────────
FEATURE_META = {
    "N":           {"label": "Nitrogen",     "unit": "kg/ha",  "icon": "🌿"},
    "P":           {"label": "Phosphorus",   "unit": "kg/ha",  "icon": "🔴"},
    "K":           {"label": "Potassium",    "unit": "kg/ha",  "icon": "🟡"},
    "temperature": {"label": "Temperature",  "unit": "°C",     "icon": "🌡️"},
    "humidity":    {"label": "Humidity",     "unit": "%",      "icon": "💧"},
    "ph":          {"label": "pH Level",     "unit": "",       "icon": "⚗️"},
    "rainfall":    {"label": "Rainfall",     "unit": "mm",     "icon": "🌧️"},
}

# ── Crop emojis ────────────────────────────────────────────────────────────
CROP_EMOJI = {
    "rice":        "🌾", "maize":       "🌽", "chickpea":    "🫘",
    "kidneybeans": "🫘", "pigeonpeas":  "🫘", "mothbeans":   "🫘",
    "mungbean":    "🫘", "blackgram":   "🫘", "lentil":      "🫘",
    "pomegranate": "🍎", "banana":      "🍌", "mango":       "🥭",
    "grapes":      "🍇", "watermelon":  "🍉", "muskmelon":   "🍈",
    "apple":       "🍎", "orange":      "🍊", "papaya":      "🍑",
    "coconut":     "🥥", "cotton":      "🪡", "jute":        "🌿",
    "coffee":      "☕",
}


class CropExplainer:
    """Wraps SHAP TreeExplainer for the Random Forest crop model."""

    def __init__(self, rf_model, scaler, label_encoder, feature_names):
        self.rf_model      = rf_model
        self.scaler        = scaler
        self.le            = label_encoder
        self.feature_names = feature_names
        # Build SHAP explainer using training-set background (fast approximation)
        self.explainer = shap.TreeExplainer(rf_model)

    def explain(self, input_values: list) -> dict:
        """
        Explain a single prediction.

        Parameters
        ----------
        input_values : list of 7 floats [N, P, K, temp, hum, ph, rain]

        Returns
        -------
        dict with keys: crop, emoji, shap_values, feature_contributions,
                        text_explanation, confidence
        """
        # Scale input
        X_input  = np.array(input_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X_input)

        # Predict
        pred_idx    = self.rf_model.predict(X_scaled)[0]
        pred_proba  = self.rf_model.predict_proba(X_scaled)[0]
        crop_name   = self.le.inverse_transform([pred_idx])[0]
        confidence  = round(float(pred_proba[pred_idx]) * 100, 2)

        # SHAP values — shape: (n_classes, n_features)
        shap_vals = self.explainer.shap_values(X_scaled)

        # Get SHAP values for the predicted class
        if isinstance(shap_vals, list):
            class_shap = shap_vals[pred_idx][0]
        else:
            class_shap = shap_vals[0, :, pred_idx]

        # Build contribution list sorted by absolute importance
        contributions = []
        for i, fname in enumerate(self.feature_names):
            sv     = float(class_shap[i])
            raw    = float(input_values[i])
            meta   = FEATURE_META.get(fname, {"label": fname, "unit": "", "icon": "📊"})
            impact = "positive" if sv > 0 else "negative"
            mag    = abs(sv)
            level  = "High" if mag > 0.3 else ("Medium" if mag > 0.1 else "Low")

            contributions.append({
                "feature":    fname,
                "label":      meta["label"],
                "unit":       meta["unit"],
                "icon":       meta["icon"],
                "value":      raw,
                "shap_value": round(sv, 4),
                "impact":     impact,
                "level":      level,
                "magnitude":  round(mag, 4),
            })

        # Sort by magnitude (largest first)
        contributions.sort(key=lambda x: x["magnitude"], reverse=True)

        # Build human-readable text explanation
        text = self._build_explanation(crop_name, contributions, confidence)

        emoji = CROP_EMOJI.get(crop_name.lower(), "🌱")

        return {
            "crop":                 crop_name,
            "emoji":                emoji,
            "confidence":           confidence,
            "shap_values":          [c["shap_value"] for c in contributions],
            "feature_labels":       [c["label"]      for c in contributions],
            "feature_contributions": contributions,
            "text_explanation":     text,
        }

    def _build_explanation(self, crop: str, contributions: list, confidence: float) -> list:
        """Build a list of readable explanation sentences."""
        sentences = []
        for c in contributions:
            direction = "supports" if c["impact"] == "positive" else "slightly reduces confidence in"
            unit      = f" {c['unit']}" if c['unit'] else ""
            if c["level"] == "High":
                sentences.append(
                    f"{c['icon']} <strong>{c['label']}</strong> value of "
                    f"<em>{c['value']:.1f}{unit}</em> strongly {direction} "
                    f"<strong>{crop.title()}</strong> recommendation."
                )
            elif c["level"] == "Medium":
                sentences.append(
                    f"{c['icon']} <strong>{c['label']}</strong> ({c['value']:.1f}{unit}) "
                    f"moderately {direction} <strong>{crop.title()}</strong>."
                )
            else:
                sentences.append(
                    f"{c['icon']} <strong>{c['label']}</strong> ({c['value']:.1f}{unit}) "
                    f"has low influence on this prediction."
                )
        return sentences
