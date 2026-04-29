# -*- coding: utf-8 -*-
"""
train.py - Crop Recommendation System
Trains Naive Bayes, Logistic Regression, and Random Forest models.
Run this script ONCE to generate the .pkl model files.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "crop_recommendation.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print(f"   CROP RECOMMENDATION - MODEL TRAINING")
print("=" * 60)

# ── Load Dataset ───────────────────────────────────────────────────────────
print(f"\n[1/6] Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"      [OK] {df.shape[0]} records, {df.shape[1]} columns loaded")
print(f"      Crops: {df['label'].nunique()} unique types")

# ── Feature / Label Split ──────────────────────────────────────────────────
X = df.drop("label", axis=1)
y = df["label"]
FEATURE_NAMES = list(X.columns)  # ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# ── Label Encoding ─────────────────────────────────────────────────────────
print("\n[2/6] Encoding labels and scaling features...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ── Feature Scaling ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)
print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Train Models ───────────────────────────────────────────────────────────
print("\n[3/6] Training models...")

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
print("      [OK] Naive Bayes - trained")

lr_model = LogisticRegression(max_iter=2000, random_state=42)
lr_model.fit(X_train, y_train)
print("      [OK] Logistic Regression - trained")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("      [OK] Random Forest - trained")

# ── Evaluate ───────────────────────────────────────────────────────────────
print("\n[4/6] Evaluating models...")
nb_pred = nb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

nb_acc = round(accuracy_score(y_test, nb_pred) * 100, 2)
lr_acc = round(accuracy_score(y_test, lr_pred) * 100, 2)
rf_acc = round(accuracy_score(y_test, rf_pred) * 100, 2)

print(f"      Naive Bayes         : {nb_acc}%")
print(f"      Logistic Regression : {lr_acc}%")
print(f"      Random Forest       : {rf_acc}%  [BEST]")

# ── Feature Importance (RF) ────────────────────────────────────────────────
importances   = rf_model.feature_importances_.tolist()
sorted_idx    = np.argsort(rf_model.feature_importances_)[::-1]
sorted_names  = [FEATURE_NAMES[i] for i in sorted_idx]
sorted_vals   = rf_model.feature_importances_[sorted_idx].tolist()

# ── Confusion Matrices ─────────────────────────────────────────────────────
cm_nb = confusion_matrix(y_test, nb_pred).tolist()
cm_lr = confusion_matrix(y_test, lr_pred).tolist()
cm_rf = confusion_matrix(y_test, rf_pred).tolist()

# ── Save Preprocessors & Models ───────────────────────────────────────────
print("\n[5/6] Saving models and preprocessors...")
joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(lr_model, os.path.join(MODEL_DIR, "lr_model.pkl"))
joblib.dump(nb_model, os.path.join(MODEL_DIR, "nb_model.pkl"))
joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le,       os.path.join(MODEL_DIR, "label_encoder.pkl"))
print("      [OK] All .pkl files saved in /models/")

# ── Save Metadata JSON ─────────────────────────────────────────────────────
print("\n[6/6] Saving metadata JSON...")
metadata = {
    "accuracies": {
        "naive_bayes":         nb_acc,
        "logistic_regression": lr_acc,
        "random_forest":       rf_acc,
    },
    "feature_names": FEATURE_NAMES,
    "feature_importances": {
        "names":  sorted_names,
        "values": sorted_vals,
        "raw":    dict(zip(FEATURE_NAMES, importances)),
    },
    "class_names": list(le.classes_),
    "confusion_matrices": {
        "naive_bayes":         cm_nb,
        "logistic_regression": cm_lr,
        "random_forest":       cm_rf,
    },
    "train_size": int(X_train.shape[0]),
    "test_size":  int(X_test.shape[0]),
    "total_records": int(df.shape[0]),
}

meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"      [OK] Metadata saved to {meta_path}")

print("\n" + "=" * 60)
print("   TRAINING COMPLETE - Ready to start the server!")
print("=" * 60)
print("\nNext step: python backend/app.py")
