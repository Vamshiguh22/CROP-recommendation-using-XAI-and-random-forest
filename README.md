# 🌾 CropAI — Explainable AI Crop Recommendation System

A full-stack web application that recommends the best crop based on soil and climate parameters, with **SHAP-powered Explainable AI** explaining every prediction.

---

## 📁 Project Structure

```
minor project/
├── backend/
│   ├── app.py          ← Flask REST API server
│   ├── train.py        ← Model training script (run once)
│   └── explainer.py    ← SHAP XAI engine
├── frontend/
│   └── index.html      ← Main UI page
├── models/             ← Auto-created after training
│   ├── rf_model.pkl
│   ├── lr_model.pkl
│   ├── nb_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── model_metadata.json
├── data/
│   └── crop_recommendation.csv
├── static/
│   ├── css/style.css
│   └── js/main.js
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Create & Activate Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Models (Run Once)

```bash
python backend/train.py
```

This generates all `.pkl` files in the `models/` folder.

### 4. Start the Server

```bash
python backend/app.py
```

### 5. Open the App

Navigate to: **http://127.0.0.1:5000**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`           | Serves the frontend UI |
| `GET`  | `/model-info` | Returns accuracies + feature importances |
| `POST` | `/predict`    | Returns predicted crop |
| `POST` | `/explain`    | Returns SHAP-based XAI explanation |

### Example `/predict` Request

```json
{
  "N": 90, "P": 42, "K": 43,
  "temperature": 20.9,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

### Example `/predict` Response

```json
{
  "status": "ok",
  "crop": "rice",
  "emoji": "🌾",
  "confidence": 96.0,
  "top5_crops": { "rice": 96.0, "jute": 2.1, ... },
  "model_predictions": {
    "random_forest":       { "crop": "rice", "confidence": 96.0 },
    "logistic_regression": { "crop": "rice", "confidence": 91.2 },
    "naive_bayes":         { "crop": "rice", "confidence": 88.5 }
  }
}
```

---

## 🤖 Models Used

| Model | Role | Typical Accuracy |
|-------|------|-----------------|
| **Random Forest** | Primary predictor + XAI | ~99% |
| Logistic Regression | Comparison | ~96% |
| Naive Bayes | Comparison | ~99% |

---

## 🧠 XAI — SHAP Explanation

SHAP (SHapley Additive exPlanations) explains **why** a crop was recommended:
- **Positive SHAP** → Feature supports this crop
- **Negative SHAP** → Feature reduces confidence in this crop
- Features are ranked by importance for every individual prediction

---

## ☁️ Deployment

### Render (Backend)

1. Push code to GitHub
2. Create a **Web Service** on [Render](https://render.com)
3. Set **Build Command**: `pip install -r requirements.txt && python backend/train.py`
4. Set **Start Command**: `python backend/app.py`

### Vercel / Netlify (Frontend only — static)

If deploying frontend separately, update the `fetch('/predict', ...)` URLs in `main.js` to point to your backend URL.

---

## 📊 Input Features

| Feature | Range | Unit |
|---------|-------|------|
| Nitrogen (N) | 0 – 200 | kg/ha |
| Phosphorus (P) | 0 – 150 | kg/ha |
| Potassium (K) | 0 – 210 | kg/ha |
| Temperature | 0 – 50 | °C |
| Humidity | 0 – 100 | % |
| pH | 0 – 14 | — |
| Rainfall | 0 – 3000 | mm |

---

## 🎓 Tech Stack

- **Backend**: Python, Flask, Flask-CORS
- **ML**: scikit-learn (RandomForest, LogisticRegression, GaussianNB)
- **XAI**: SHAP (TreeExplainer)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Charts**: Chart.js 4
- **Fonts**: Inter + Space Grotesk (Google Fonts)
